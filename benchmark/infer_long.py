"""Compare Qwen-base vs SuperCoder vs Lazy+Morph on the CodeNet balanced dataset,
bucketed by C code length (short / medium_short / medium_long / long).

For each sample the container compiles the C code to baseline x86-64 assembly
with `gcc -S -O3` (matches the SuperCoder paper baseline; the data field
`unoptimized_assembly` in supercoder parquets is also -O3 output despite the
misleading name), builds the model prompt, generates optimized assembly, then
evaluates correctness and optionally measures speedup against the same -O3
binary.

Usage
-----
    cd qwen-long-context/benchmark
    modal run infer_long.py                              # 50 per bucket, correctness only
    modal run infer_long.py --do-speedup                 # include speedup timing
    modal run infer_long.py --n-per-bucket 20            # quick smoke test
    modal run infer_long.py --temperature 0.5

Output (written locally)
------------------------
    long_results/infer_results_<tag>.csv      per-sample detail per model
    long_results/infer_summary_by_bucket.csv  per-bucket per-model metrics
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import modal

MINUTES  = 60
HERE     = Path(__file__).resolve().parent
REPO     = (HERE / "..").resolve()
REWARD   = (REPO / "training/train1-lazy-supercoder/reward.py").resolve()
VERL_DIR = (REPO / "training/verl").resolve()
DATA_DIR = (REPO / "experiments/1-experiment-supercoder-long/data/codenet_balanced_hf").resolve()

QWEN_BASE = "Qwen/Qwen2.5-Coder-7B-Instruct"
EXP1_DEFAULT_CKPT = "/checkpoints/exp1-train-supercoder/global_step_420/hf_model"
LAZY_EXP  = "train2-lazy-supercoder"

BUCKET_ORDER = ["short", "medium_short", "medium_long", "long"]

# Original SuperCoder prompt (for qwen-base and supercoder)
ORIGINAL_PROMPT_TEMPLATE = (
    "Given the following C code and assembly code, your task is to generate "
    "highly optimized x86-64 assembly code.\n"
    "C Code:\n\n"
    "```c\n{c_code}\n```\n\n"
    "Assembly Code:\n\n"
    "```assembly\n{unopt_asm}\n```\n\n"
    "Only output the optimized assembly code. Do not include any other text. "
    "Do not write any comments in the assembly code. Wrap the assembly code "
    "in ```assembly``` tags.\nOptimized Assembly Code:\n"
)

# Lazy edit prompt (for lazy-morph model)
LAZY_PROMPT_TEMPLATE = (
    "Given the following C code and assembly code, your task is to generate "
    "highly optimized x86-64 assembly code.\n"
    "C Code:\n\n"
    "```c\n{c_code}\n```\n\n"
    "Assembly Code:\n\n"
    "```assembly\n{unopt_asm}\n```\n\n"
    "Only output the lazy edit update in this exact format:\n"
    "```assembly\n"
    "<lazy edit assembly update>\n"
    "```\n\n"
    "How to lazy edit:\n"
    'Use "// ... existing code ..." to represent unchanged code blocks. '
    "Include just enough surrounding context to locate each edit precisely.\n\n"
    "Example format:\n"
    "// ... existing code ...\n"
    "FIRST_EDIT\n"
    "// ... existing code ...\n"
    "SECOND_EDIT\n"
    "// ... existing code ...\n"
    "\n"
    "Rules:\n"
    '- ALWAYS use "// ... existing code ..." for unchanged sections (omitting this marker will cause deletions)\n'
    "- Include minimal context around edits only when needed for disambiguation\n"
    "- Preserve exact indentation\n"
    "- For deletions: show context before and after, omit the deleted lines\n"
    "- Batch multiple edits to the same file in one call\n"
    "\nOptimized lazy edit assembly update:\n"
)

app = modal.App("infer-long-context-benchmark")

hf_secret       = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])
morph_secret    = modal.Secret.from_name("morph", required_keys=["MORPH_API_KEY"])
hf_cache_vol    = modal.Volume.from_name("huggingface-cache",    create_if_missing=True)
vllm_cache_vol  = modal.Volume.from_name("vllm-cache",           create_if_missing=True)
checkpoints_vol = modal.Volume.from_name("debug-rl-checkpoints", create_if_missing=False)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("gcc", "time", "curl", "ca-certificates")
    .run_commands(
        "curl -sL https://github.com/sharkdp/hyperfine/releases/download/v1.18.0/"
        "hyperfine-v1.18.0-x86_64-unknown-linux-gnu.tar.gz "
        "| tar xz -C /tmp && "
        "install -m 0755 /tmp/hyperfine-v1.18.0-x86_64-unknown-linux-gnu/hyperfine "
        "/usr/local/bin/hyperfine && hyperfine --version",
    )
    .add_local_dir(str(VERL_DIR), "/verl_src", copy=True)
    .run_commands(
        "pip install torch==2.6.0 torchaudio==2.6.0 torchdata==0.11.0 torchvision==0.21.0"
        " tabulate fire 'ray[default]' psutil cachetools numpy pandas pyarrow openai",
        "pip install wheel && pip install flash-attn==2.7.4.post1 --no-build-isolation",
        "pip install -e '/verl_src[vllm]'",
        "pip install 'transformers>=4.40,<5'",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_file(str(REWARD), "/reward.py", copy=True)
    .add_local_dir(str(DATA_DIR), "/data/codenet_balanced_hf", copy=True)
)


# -- Checkpoint helper --------------------------------------------------------

def _find_hf_checkpoint(exp_name: str, step: int | None = None) -> str | None:
    import os, re
    root = f"/checkpoints/{exp_name}"
    if not os.path.isdir(root):
        return None
    step_dirs = sorted(
        [(int(m.group(1)), os.path.join(root, name))
         for name in os.listdir(root)
         if (m := re.match(r"global_step_(\d+)$", name))],
        key=lambda x: x[0], reverse=True,
    )
    if not step_dirs:
        return None
    if step is not None:
        step_dirs = [(s, d) for s, d in step_dirs if s == step]
    for _step, step_dir in step_dirs:
        for sub in ("hf_model", "actor/huggingface", "actor/hf", "actor", ""):
            candidate = os.path.join(step_dir, sub) if sub else step_dir
            if os.path.isfile(os.path.join(candidate, "config.json")):
                print(f"  [ckpt] using: {candidate}")
                return candidate
    return None


def _extract_assembly(raw: str) -> str:
    text = raw
    if "```assembly" in text:
        text = text[text.rfind("```assembly") + len("```assembly"):]
    elif "```asm" in text:
        text = text[text.rfind("```asm") + len("```asm"):]
    elif text.startswith("```"):
        text = text[3:]
    if "```" in text:
        text = text[:text.rfind("```")]
    return text.strip()


# -- Single-model Modal function ----------------------------------------------

@app.function(
    image=image,
    gpu="h100:1",
    timeout=180 * MINUTES,
    secrets=[hf_secret, morph_secret],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm":        vllm_cache_vol,
        "/checkpoints":             checkpoints_vol,
    },
)
def run_model(
    tag: str,
    model_path: str,
    exp_name: str = "",
    exp_step: int = 0,
    is_lazy: bool = False,
    samples: list[dict] | None = None,
    temperature: float = 0.2,
    max_model_len: int = 16384,
    max_tokens: int = 4096,
    do_speedup: bool = False,
    max_inputs_for_speedup: int = 3,
) -> tuple[str, list[dict]]:
    import gc, json, os, subprocess, sys, tempfile, time
    import numpy as np
    import vllm
    from transformers import AutoTokenizer
    sys.path.insert(0, "/")
    from reward import (
        check_correctness, prepare_solution_assembly,
        strip_assembly_fence,
    )

    # Resolve checkpoint
    if exp_name and not model_path:
        step = exp_step if exp_step > 0 else None
        model_path = _find_hf_checkpoint(exp_name, step=step) or ""
    if not model_path:
        print(f"[{tag}] No model path -- skipping.")
        return tag, []

    print(f"\n[{tag}] model={model_path}  samples={len(samples)}  lazy={is_lazy}")

    # Compile C -> unoptimized assembly for each sample
    for s in samples:
        with tempfile.TemporaryDirectory() as d:
            c_file = os.path.join(d, "x.c")
            s_file = os.path.join(d, "x.s")
            with open(c_file, "w") as f:
                f.write(s["c_code"])
            r = subprocess.run(
                ["gcc", "-S", "-O3", "-o", s_file, c_file],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode != 0 or not os.path.exists(s_file):
                s["unopt_asm"] = ""
                s["compile_unopt_ok"] = False
                continue
            with open(s_file, "r") as f:
                s["unopt_asm"] = f.read()
            s["compile_unopt_ok"] = True

    ok_samples = [s for s in samples if s.get("compile_unopt_ok")]
    print(f"[{tag}] compiled unopt OK: {len(ok_samples)}/{len(samples)}")

    # Build prompts. Drop rows whose tokenized prompt would exceed the model
    # context (max_model_len - max_tokens) so vLLM doesn't crash.
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt_budget = max(0, max_model_len - max_tokens)
    prompts = []
    overlong_count = 0
    for s in samples:
        if not s.get("compile_unopt_ok"):
            prompts.append("")
            s["prompt_tokens"] = 0
            s["overlong"] = False
            continue
        if is_lazy:
            content = LAZY_PROMPT_TEMPLATE.format(c_code=s["c_code"], unopt_asm=s["unopt_asm"])
        else:
            content = ORIGINAL_PROMPT_TEMPLATE.format(c_code=s["c_code"], unopt_asm=s["unopt_asm"])
        msgs = [{"role": "user", "content": content}]
        chat_prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = len(tokenizer(chat_prompt, add_special_tokens=False)["input_ids"])
        s["prompt_tokens"] = prompt_tokens
        s["overlong"] = prompt_tokens > prompt_budget
        if s["overlong"]:
            overlong_count += 1
            prompts.append("")
        else:
            prompts.append(chat_prompt)

    print(f"[{tag}] Filtered overlong prompts: {overlong_count}/{len(samples)} "
          f"(prompt_budget={prompt_budget} tokens)")

    llm = vllm.LLM(
        model=model_path,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
    )
    sp = vllm.SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop_token_ids=[151643, 151645],
    )

    gen_idx = [i for i, p in enumerate(prompts) if p]
    gen_prompts = [prompts[i] for i in gen_idx]
    print(f"[{tag}] generating {len(gen_prompts)} / {len(prompts)}")
    t0 = time.time()
    outputs = llm.generate(gen_prompts, sp) if gen_prompts else []
    print(f"[{tag}] generate done in {time.time()-t0:.1f}s")

    out_by_pos: dict[int, tuple[str, int]] = {}
    for out, i in zip(outputs, gen_idx):
        out_by_pos[i] = (out.outputs[0].text, len(out.outputs[0].token_ids))

    # Evaluate each sample
    results = []
    for idx, s in enumerate(samples):
        raw_response, n_tok = out_by_pos.get(idx, ("", 0))

        if not s.get("compile_unopt_ok") or s.get("overlong") or not raw_response:
            if not s.get("compile_unopt_ok"):
                status = "COMPILE_UNOPT_FAIL"
            elif s.get("overlong"):
                status = "PROMPT_TOO_LONG"
            else:
                status = "NO_RESPONSE"
            results.append({
                "idx": idx + 1,
                "bucket": s["bucket"],
                "problem_id": s.get("problem_id", ""),
                "c_loc": s.get("c_loc", 0),
                "prompt_tokens": s.get("prompt_tokens", 0),
                "overlong": bool(s.get("overlong", False)),
                "status": status,
                "compiled": False,
                "tests_pass": False,
                "correctness": -1,
                "raw_speedup": None,
                "effective_speedup": 1.0,
                "speedup_floor1": 1.0,
                "is_copy": False,
                "response_tokens": n_tok,
                "morph_called": False if is_lazy else None,
                "morph_success": False if is_lazy else None,
            })
            continue

        # For lazy-morph: merge via Morph API
        morph_called = False
        morph_success = False
        extra_info = {"unoptimized_assembly": s["unopt_asm"]}

        if is_lazy:
            asm, morph_metrics = prepare_solution_assembly(raw_response, extra_info)
            morph_called = morph_metrics.get("morph/called", 0) > 0
            morph_success = morph_metrics.get("morph/success", 0) > 0
            if asm is None:
                asm = ""
        else:
            asm = _extract_assembly(raw_response)

        # Parse test cases
        try:
            test_cases = json.loads(s.get("test_cases", "[]"))
        except Exception:
            test_cases = []
        tests = test_cases[:10]

        # Compile generated assembly
        compiled = False
        tests_pass = False
        n_pass = 0
        compile_stderr = ""
        gen_binary = None

        if asm:
            with tempfile.TemporaryDirectory() as d:
                asm_f = os.path.join(d, "gen.s")
                bin_f = os.path.join(d, "gen.bin")
                with open(asm_f, "w") as f:
                    f.write(asm)
                cr = subprocess.run(
                    f"gcc {asm_f} -o {bin_f} -lm",
                    shell=True, capture_output=True, text=True, timeout=30,
                )
                if cr.returncode == 0 and os.path.exists(bin_f):
                    compiled = True
                    with open(bin_f, "rb") as f:
                        gen_binary = f.read()
                else:
                    compile_stderr = cr.stderr[:400]

        # Run tests
        if compiled and gen_binary and tests:
            with tempfile.TemporaryDirectory() as d:
                bin_f = os.path.join(d, "gen.bin")
                with open(bin_f, "wb") as f:
                    f.write(gen_binary)
                os.chmod(bin_f, 0o755)

                for t in tests:
                    try:
                        r = subprocess.run(
                            [bin_f], input=t.get("input", ""),
                            capture_output=True, text=True, timeout=10,
                        )
                        if r.returncode == 0 and r.stdout.rstrip() == (t.get("output", "") or "").rstrip():
                            n_pass += 1
                    except Exception:
                        pass

            tests_pass = (len(tests) > 0 and n_pass == len(tests))

        if not compiled:       correctness = -1.0
        elif not tests:        correctness = -1.0
        elif tests_pass:       correctness = 1.0
        else:                  correctness = n_pass / len(tests) if tests else 0.0

        if correctness == -1:       status = "COMPILE_FAIL"
        elif correctness == -0.5:   status = "RUNTIME_ERR"
        elif correctness == 1.0:    status = "ALL_PASS"
        else:                       status = f"PARTIAL_{correctness:.0%}"

        is_copy = (asm.strip() == s["unopt_asm"].strip()) if asm else False

        # Speedup
        raw_speedup = None
        if do_speedup and tests_pass and gen_binary:
            try:
                with tempfile.TemporaryDirectory() as d:
                    gen_bin = os.path.join(d, "gen.bin")
                    unopt_bin = os.path.join(d, "unopt.bin")
                    unopt_s = os.path.join(d, "unopt.s")
                    with open(gen_bin, "wb") as f: f.write(gen_binary)
                    os.chmod(gen_bin, 0o755)
                    with open(unopt_s, "w") as f: f.write(s["unopt_asm"])
                    cr = subprocess.run(
                        f"gcc {unopt_s} -o {unopt_bin} -lm",
                        shell=True, capture_output=True, text=True, timeout=30,
                    )
                    if cr.returncode == 0:
                        os.chmod(unopt_bin, 0o755)
                        speedups = []
                        for t in tests[:max_inputs_for_speedup]:
                            inf = os.path.join(d, "inp.txt")
                            out_j = os.path.join(d, "bench.json")
                            with open(inf, "w") as f: f.write(t.get("input", ""))

                            def _bench(bp, out_j=out_j, inf=inf):
                                r = subprocess.run(
                                    ["hyperfine", "--warmup", "3", "--runs", "10",
                                     "--input", inf, "--export-json", out_j,
                                     "--time-unit", "millisecond", bp],
                                    capture_output=True, text=True, timeout=60,
                                )
                                if r.returncode != 0 or not os.path.exists(out_j):
                                    return None
                                try:
                                    return json.load(open(out_j))["results"][0]["mean"] * 1000
                                except Exception:
                                    return None

                            u_ms = _bench(unopt_bin)
                            g_ms = _bench(gen_bin)
                            if u_ms and g_ms and g_ms > 0:
                                speedups.append(u_ms / g_ms)
                        if speedups:
                            raw_speedup = float(np.mean(speedups))
            except Exception as e:
                print(f"[{tag}] speedup exception: {e}")

        effective_speedup = raw_speedup if (tests_pass and raw_speedup is not None) else 1.0
        speedup_floor1 = max(1.0, effective_speedup)

        sp_str = f"speedup={raw_speedup:.3f}x" if raw_speedup is not None else "speedup=N/A"
        morph_str = f"  morph={'OK' if morph_success else 'FAIL'}" if morph_called else ""
        print(f"[{tag}] [{idx+1:3d}/{len(samples)}] {s['bucket']:<14s} {status:<20s} {sp_str} copy={'Y' if is_copy else 'N'}{morph_str}")

        results.append({
            "idx": idx + 1,
            "bucket": s["bucket"],
            "problem_id": s.get("problem_id", ""),
            "c_loc": s.get("c_loc", 0),
            "prompt_tokens": s.get("prompt_tokens", 0),
            "overlong": bool(s.get("overlong", False)),
            "status": status,
            "compiled": compiled,
            "tests_pass": tests_pass,
            "correctness": round(float(correctness), 3),
            "n_tests": len(tests),
            "n_tests_pass": n_pass,
            "raw_speedup": round(raw_speedup, 4) if raw_speedup is not None else None,
            "effective_speedup": round(float(effective_speedup), 4),
            "speedup_floor1": round(float(speedup_floor1), 4),
            "is_copy": is_copy,
            "response_tokens": n_tok,
            "compile_stderr": compile_stderr,
            "morph_called": morph_called if is_lazy else None,
            "morph_success": morph_success if is_lazy else None,
        })

    # Summary
    n = len(results)
    print(f"\n[{tag}] Compile OK : {sum(r['compiled'] for r in results)}/{n}")
    print(f"[{tag}] All Pass   : {sum(r['tests_pass'] for r in results)}/{n}")
    for b in BUCKET_ORDER:
        br = [r for r in results if r["bucket"] == b]
        if br:
            print(f"[{tag}]   {b:<14s} compile={sum(r['compiled'] for r in br)}/{len(br)}  "
                  f"pass={sum(r['tests_pass'] for r in br)}/{len(br)}")

    del llm; gc.collect()
    return tag, results


# -- Local entrypoint ---------------------------------------------------------

@app.local_entrypoint()
def main(
    n_per_bucket: int          = 50,
    random_seed: int           = 42,
    temperature: float         = 0.2,
    max_model_len: int         = 16384,
    max_tokens: int            = 4096,
    do_speedup: bool           = False,
    supercoder_ckpt: str       = "",
    lazy_morph_ckpt: str       = "",
    lazy_morph_step: int       = 0,
    out_dir: str               = "",
) -> None:
    import random
    import numpy as np
    from datasets import load_from_disk

    out_path = Path(out_dir) if out_dir else (HERE / "long_results")
    out_path.mkdir(parents=True, exist_ok=True)

    # Load and sample dataset
    ds = load_from_disk(str(DATA_DIR))
    print(f"Loaded dataset: {len(ds)} rows, columns={ds.column_names}")

    per_bucket: dict[str, list[int]] = {b: [] for b in BUCKET_ORDER}
    for i, b in enumerate(ds["bucket"]):
        if b in per_bucket:
            per_bucket[b].append(i)

    rng = random.Random(random_seed)
    sampled_idxs: list[int] = []
    for b in BUCKET_ORDER:
        pool = per_bucket[b]
        take = min(n_per_bucket, len(pool))
        if take < n_per_bucket:
            print(f"WARN bucket={b}: only {len(pool)} available (< {n_per_bucket})")
        sampled_idxs.extend(rng.sample(pool, take))

    print(f"Total samples: {len(sampled_idxs)} "
          f"(target {n_per_bucket * len(BUCKET_ORDER)})")

    samples: list[dict] = []
    for i in sampled_idxs:
        row = ds[i]
        samples.append({
            "bucket":     row["bucket"],
            "problem_id": row["problem_id"],
            "c_code":     row["c_code"],
            "c_loc":      row["c_loc"],
            "test_cases": row["test_cases"],
        })

    # Spawn 3 models
    sc_path = supercoder_ckpt or EXP1_DEFAULT_CKPT
    shared = dict(
        samples=samples, temperature=temperature,
        max_model_len=max_model_len, max_tokens=max_tokens,
        do_speedup=do_speedup,
    )

    print("Spawning 3 model containers in parallel...")
    calls = [
        run_model.spawn(tag="qwen-base", model_path=QWEN_BASE, is_lazy=False, **shared),
        run_model.spawn(tag="supercoder", model_path=sc_path, is_lazy=False, **shared),
        run_model.spawn(tag="lazy-morph", model_path=lazy_morph_ckpt, exp_name=LAZY_EXP, exp_step=lazy_morph_step, is_lazy=True, **shared),
    ]

    all_results: dict[str, list[dict]] = {}
    for call in calls:
        tag, results = call.get()
        if results:
            all_results[tag] = results
            print(f"  [{tag}] finished -- {len(results)} samples")
        else:
            print(f"  [{tag}] skipped (no checkpoint found)")

    # Write per-model detail CSVs
    detail_fields = [
        "idx", "bucket", "problem_id", "c_loc", "prompt_tokens", "overlong",
        "status", "compiled", "tests_pass",
        "correctness", "n_tests", "n_tests_pass",
        "raw_speedup", "effective_speedup", "speedup_floor1",
        "is_copy", "response_tokens", "compile_stderr",
        "morph_called", "morph_success",
    ]
    for tag, results in all_results.items():
        csv_path = out_path / f"infer_results_{tag}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=detail_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved {len(results)} rows -> {csv_path}")

    # Write summary by bucket
    summary_rows = []
    for tag, results in all_results.items():
        for b in BUCKET_ORDER:
            br = [r for r in results if r["bucket"] == b]
            if not br:
                continue
            n = len(br)
            sp_vals     = [r["raw_speedup"] for r in br if r["raw_speedup"] is not None]
            floor1_vals = [r["speedup_floor1"] for r in br]
            geo_floor1  = float(np.exp(np.mean(np.log(floor1_vals)))) if floor1_vals else 1.0

            summary_rows.append({
                "model":              tag,
                "bucket":             b,
                "n":                  n,
                "compile_rate":       round(sum(r["compiled"] for r in br) / n, 3),
                "test_pass_rate":     round(sum(r["tests_pass"] for r in br) / n, 3),
                "mean_correctness":   round(sum(r["correctness"] for r in br) / n, 3),
                "n_speedup_measured": len(sp_vals),
                "mean_speedup":       round(float(np.mean(sp_vals)), 4) if sp_vals else "",
                "geo_mean_floor1":    round(max(1.0, geo_floor1), 4),
                "copy_rate":          round(sum(r["is_copy"] for r in br) / n, 3),
            })

    summary_path = out_path / "infer_summary_by_bucket.csv"
    summary_fields = [
        "model", "bucket", "n", "compile_rate", "test_pass_rate",
        "mean_correctness", "n_speedup_measured", "mean_speedup",
        "geo_mean_floor1", "copy_rate",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary -> {summary_path}")

    # Pretty-print
    col_w = 14
    models = list(all_results.keys())
    print(f"\n{'='*90}")
    print(f"LONG-CONTEXT BENCHMARK  (n_per_bucket={n_per_bucket}, temp={temperature})")
    print(f"{'='*90}")
    for b in BUCKET_ORDER:
        print(f"\n--- {b} ---")
        hdr = f"{'metric':<22}" + "".join(f"{m:>{col_w}}" for m in models)
        print(hdr)
        print("-" * len(hdr))
        for key in ["compile_rate", "test_pass_rate", "mean_correctness", "mean_speedup", "geo_mean_floor1", "copy_rate"]:
            row_str = f"{key:<22}"
            for m in models:
                row = next((r for r in summary_rows if r["model"] == m and r["bucket"] == b), {})
                row_str += f"{str(row.get(key, '')):>{col_w}}"
            print(row_str)
    print()
