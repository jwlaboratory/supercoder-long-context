"""Compare Qwen-base vs SuperCoder (exp1) vs Lazy+Morph (train2-lazy-supercoder)
on the SuperCoder validation set (``sc_val.parquet``).

Each model runs in its own GPU container in parallel. The lazy-morph model
generates lazy edit patches and uses the Morph API to merge them into the
original assembly before evaluation.

Usage
-----
    cd qwen-long-context/benchmark
    modal run infer.py                        # 200 samples, correctness only
    modal run infer.py --do-speedup           # include hyperfine speedup
    modal run infer.py --n-samples 50         # quick smoke test
    modal run infer.py --temperature 0.5
    modal run infer.py --lazy-morph-step 300  # evaluate a specific lazy checkpoint

Checkpoint paths (inside the ``debug-rl-checkpoints`` Modal volume):
    --supercoder-ckpt    default: exp1-train-supercoder/global_step_420/hf_model
    --lazy-morph-ckpt    default: auto-detect latest train2-lazy-supercoder step
    --lazy-morph-step    default: 0 (latest); set 300 for global_step_300

Merge before running (if not done):
    modal run merge_checkpoint.py --step 300

Output (written locally)
------------------------
    infer_results_<tag>.csv    per-sample detail per model
    infer_summary.csv          side-by-side metric comparison
"""
from __future__ import annotations

import csv
from pathlib import Path
import modal

MINUTES = 60
HERE    = Path(__file__).resolve().parent
REPO    = (HERE / "..").resolve()
REWARD  = (REPO / "training/train1-lazy-supercoder/reward.py").resolve()
VERL_DIR = (REPO / "training/verl").resolve()

QWEN_BASE = "Qwen/Qwen2.5-Coder-7B-Instruct"
EXP1_NAME = "exp1-train-supercoder"
EXP1_DEFAULT_CKPT = f"/checkpoints/{EXP1_NAME}/global_step_420/hf_model"
LAZY_EXP  = "train2-lazy-supercoder"

# Original SuperCoder prompt (for qwen-base and supercoder)
ORIGINAL_PROMPT_TEMPLATE = (
    "You are an expert x86-64 assembly programmer. "
    "Given the following C code and its unoptimized assembly, "
    "generate a highly optimized x86-64 assembly version that produces "
    "the same output for all inputs.\n\n"
    "C Code:\n\n```c\n{c_code}\n```\n\n"
    "Assembly Code:\n\n```assembly\n{unopt_asm}\n```\n\n"
    "Generated, optimized assembly:\n"
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

app = modal.App("infer-lazy-supercoder-benchmark")

hf_secret       = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])
morph_secret    = modal.Secret.from_name("morph", required_keys=["MORPH_API_KEY"])
data_vol        = modal.Volume.from_name("debug-rl-data",        create_if_missing=False)
checkpoints_vol = modal.Volume.from_name("debug-rl-checkpoints", create_if_missing=False)
hf_cache_vol    = modal.Volume.from_name("huggingface-cache",    create_if_missing=True)
vllm_cache_vol  = modal.Volume.from_name("vllm-cache",           create_if_missing=True)

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
    .add_local_file(str(REWARD), "/reward.py")
)


# -- Checkpoint helper (runs inside container) --------------------------------

def _find_hf_checkpoint(exp_name: str, step: int | None = None) -> str | None:
    """Find a merged HF checkpoint under /checkpoints/<exp_name>/global_step_*."""
    import os, re

    root = f"/checkpoints/{exp_name}"
    if not os.path.isdir(root):
        print(f"  [ckpt] not found: {root}")
        return None

    step_dirs = sorted(
        [(int(m.group(1)), os.path.join(root, name))
         for name in os.listdir(root)
         if (m := re.match(r"global_step_(\d+)$", name))],
        key=lambda x: x[0], reverse=True,
    )
    if not step_dirs:
        print(f"  [ckpt] no global_step_N dirs in {root}")
        return None

    if step is not None:
        step_dirs = [(s, d) for s, d in step_dirs if s == step]
        if not step_dirs:
            print(f"  [ckpt] step {step} not found in {root}")
            return None

    print(f"  [ckpt] steps considered: {[s for s, _ in step_dirs]}")
    for _step, step_dir in step_dirs:
        for sub in ("hf_model", "actor/huggingface", "actor/hf", "actor", ""):
            candidate = os.path.join(step_dir, sub) if sub else step_dir
            if os.path.isfile(os.path.join(candidate, "config.json")):
                print(f"  [ckpt] using: {candidate}")
                return candidate

    print(f"  [ckpt] no config.json found under {root}")
    return None


def _strip_assembly_fence(text):
    """Strip surrounding assembly/code fence."""
    text = (text or "").strip()
    if "```assembly" in text:
        text = text[text.rfind("```assembly") + len("```assembly"):]
    elif "```asm" in text:
        text = text[text.rfind("```asm") + len("```asm"):]
    elif text.startswith("```"):
        text = text[3:]
    if "```" in text:
        text = text[:text.rfind("```")]
    return text.strip()


# -- Single-model Modal function (one GPU container per model) ----------------

@app.function(
    image=image,
    gpu="h100:1",
    timeout=120 * MINUTES,
    secrets=[hf_secret, morph_secret],
    volumes={
        "/data":                    data_vol,
        "/checkpoints":             checkpoints_vol,
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm":        vllm_cache_vol,
    },
)
def eval_model(
    tag: str,
    model_path: str,
    exp_name: str = "",
    exp_step: int = 0,
    is_lazy: bool = False,
    n_samples: int = 200,
    parquet: str = "sc_val",
    temperature: float = 0.0,
    random_seed: int = 42,
    do_speedup: bool = False,
    max_inputs_for_speedup: int = 10,
) -> tuple[str, list[dict]]:
    import gc, json, os, subprocess, sys, tempfile, time
    import numpy as np
    import pandas as pd
    import vllm
    from transformers import AutoTokenizer
    sys.path.insert(0, "/")
    from reward import check_correctness, prepare_solution_assembly

    # Resolve checkpoint if needed
    if exp_name and not model_path:
        step = exp_step if exp_step > 0 else None
        model_path = _find_hf_checkpoint(exp_name, step=step) or ""
    if not model_path:
        print(f"[{tag}] No model path -- skipping.")
        return tag, []

    # Load the same sample every time (fixed seed)
    df      = pd.read_parquet(f"/data/{parquet}.parquet")
    samples = [row for _, row in df.sample(n=min(n_samples, len(df)), random_state=random_seed).iterrows()]
    print(f"\n[{tag}] model={model_path}  parquet={parquet}  samples={len(samples)}  lazy={is_lazy}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = vllm.LLM(
        model=model_path,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )
    sampling_params = vllm.SamplingParams(
        temperature=temperature,
        max_tokens=2000,
        stop_token_ids=[151643, 151645],
    )

    # Build prompts from extra_info (not stored prompts) so all models
    # get the correct prompt format regardless of parquet version.
    prompts = []
    for row in samples:
        ei = row["extra_info"] if isinstance(row["extra_info"], dict) else {}
        c_code   = ei.get("c_code", "")
        unopt_asm = _strip_assembly_fence(ei.get("unoptimized_assembly", ""))

        if is_lazy:
            content = LAZY_PROMPT_TEMPLATE.format(c_code=c_code, unopt_asm=unopt_asm)
        else:
            content = ORIGINAL_PROMPT_TEMPLATE.format(c_code=c_code, unopt_asm=unopt_asm)

        msgs = [{"role": "user", "content": content}]
        prompts.append(
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        )

    print(f"[{tag}] Generating {len(prompts)} responses ...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    print(f"[{tag}] Done in {time.time()-t0:.1f}s")

    results = []
    for i, (output, row) in enumerate(zip(outputs, samples)):
        raw_response    = output.outputs[0].text
        response_tokens = len(output.outputs[0].token_ids)

        ei           = row["extra_info"] if isinstance(row["extra_info"], dict) else {}
        ground_truth = row["reward_model"].get("ground_truth", "") if isinstance(row["reward_model"], dict) else ""

        # For lazy-morph: merge the edit via Morph API, then evaluate
        morph_called  = False
        morph_success = False
        if is_lazy:
            asm, morph_metrics = prepare_solution_assembly(raw_response, ei)
            morph_called  = morph_metrics.get("morph/called", 0) > 0
            morph_success = morph_metrics.get("morph/success", 0) > 0
            if asm is None:
                # Morph merge failed
                asm = ""
        else:
            asm = _strip_assembly_fence(raw_response)

        # Compile stderr capture
        compile_stderr = ""
        try:
            with tempfile.TemporaryDirectory() as d:
                asm_f = os.path.join(d, "sol.s")
                bin_f = os.path.join(d, "sol.bin")
                with open(asm_f, "w") as f:
                    f.write(asm)
                cr = subprocess.run(
                    f"gcc {asm_f} -o {bin_f} -lm",
                    shell=True, capture_output=True, text=True, timeout=30,
                )
                compile_stderr = cr.stderr[:400] if cr.returncode != 0 else ""
        except Exception as e:
            compile_stderr = str(e)[:200]

        correctness, binary = check_correctness(asm, ground_truth, ei)
        compiled   = (correctness != -1)
        tests_pass = (correctness == 1.0)

        if correctness == -1:       status = "COMPILE_FAIL"
        elif correctness == -0.5:   status = "RUNTIME_ERR"
        elif correctness == 1.0:    status = "ALL_PASS"
        else:                       status = f"PARTIAL_{correctness:.0%}"

        unopt_stripped = _strip_assembly_fence(ei.get("unoptimized_assembly", ""))
        is_copy = (asm.strip() == unopt_stripped)

        # Speedup (only for ALL_PASS)
        raw_speedup = None
        if do_speedup and tests_pass and binary is not None:
            _inputs  = list(ei.get("inputs", []))[:max_inputs_for_speedup]
            _precomp = bytes(ei.get("unoptimized_compiled", b""))
            if _inputs and _precomp:
                try:
                    with tempfile.TemporaryDirectory() as d:
                        sol_bin   = os.path.join(d, "sol.bin")
                        unopt_bin = os.path.join(d, "unopt.bin")
                        with open(sol_bin,   "wb") as f: f.write(binary)
                        with open(unopt_bin, "wb") as f: f.write(_precomp)
                        os.chmod(sol_bin, 0o755); os.chmod(unopt_bin, 0o755)

                        speedups = []
                        for j, inp_text in enumerate(_inputs):
                            inf   = os.path.join(d, f"in{j}.txt")
                            out_j = os.path.join(d, f"b{j}.json")
                            with open(inf, "w") as f: f.write(inp_text)

                            def _bench(bp, out_j=out_j, inf=inf):
                                r = subprocess.run(
                                    [
                                        "hyperfine",
                                        "--warmup", "3",
                                        "--runs", "10",
                                        "--input", inf,
                                        "--export-json", out_j,
                                        "--time-unit", "millisecond",
                                        bp,
                                    ],
                                    capture_output=True, text=True, timeout=60,
                                )
                                if r.returncode != 0 or not os.path.exists(out_j):
                                    return None
                                try:
                                    return json.load(open(out_j))["results"][0]["mean"] * 1000
                                except Exception:
                                    return None

                            u_ms = _bench(unopt_bin)
                            s_ms = _bench(sol_bin)
                            if u_ms and s_ms and s_ms > 0:
                                speedups.append(u_ms / s_ms)

                        if speedups:
                            raw_speedup = float(np.mean(speedups))
                except Exception as e:
                    print(f"[{tag}] speedup exception: {e}")

        effective_speedup = raw_speedup if (tests_pass and raw_speedup is not None) else 1.0
        speedup_floor1 = max(1.0, effective_speedup)

        sp_str = f"speedup={raw_speedup:.3f}x" if raw_speedup is not None else "speedup=N/A"
        morph_str = f"  morph={'OK' if morph_success else 'FAIL'}" if morph_called else ""
        print(f"[{tag}] [{i+1:3d}/{len(samples)}]  {status:<20s}  {sp_str}  copy={'Y' if is_copy else 'N'}{morph_str}")

        row_result = {
            "idx":               i + 1,
            "problem_idx":       ei.get("problem_idx", -1),
            "status":            status,
            "compiled":          compiled,
            "tests_pass":        tests_pass,
            "correctness":       round(float(correctness), 3),
            "raw_speedup":       round(raw_speedup, 4) if raw_speedup is not None else None,
            "effective_speedup": round(float(effective_speedup), 4),
            "speedup_floor1":    round(float(speedup_floor1), 4),
            "is_copy":           is_copy,
            "response_tokens":   response_tokens,
            "n_inputs":          len(ei.get("inputs", [])),
            "compile_stderr":    compile_stderr,
            "response":          raw_response,
        }
        if is_lazy:
            row_result["morph_called"]  = morph_called
            row_result["morph_success"] = morph_success
        results.append(row_result)

    n = len(results)
    sp_vals = [r["raw_speedup"] for r in results if r["raw_speedup"] is not None]
    eff_vals = [r["effective_speedup"] for r in results]
    print(f"\n[{tag}] Compile OK : {sum(r['compiled']   for r in results)}/{n}")
    print(f"[{tag}] All Pass   : {sum(r['tests_pass'] for r in results)}/{n}")
    print(f"[{tag}] Mean corr  : {sum(r['correctness'] for r in results)/n:.3f}")
    if sp_vals:
        print(f"[{tag}] Speedup    : mean={np.mean(sp_vals):.3f}x  max={max(sp_vals):.3f}x (ALL_PASS only)")
    print(f"[{tag}] Eff speedup: mean={np.mean(eff_vals):.3f}x (failures count as 1.0x)")
    if is_lazy:
        n_morph = sum(1 for r in results if r.get("morph_called"))
        n_morph_ok = sum(1 for r in results if r.get("morph_success"))
        print(f"[{tag}] Morph      : called={n_morph}/{n}  success={n_morph_ok}/{n_morph if n_morph else 1}")

    del llm; gc.collect()
    return tag, results


# -- Local entrypoint -- spawns all 3 containers in parallel ------------------

@app.local_entrypoint()
def main(
    n_samples: int             = 200,
    parquet: str               = "sc_val",
    temperature: float         = 0.0,
    random_seed: int           = 42,
    do_speedup: bool           = False,
    supercoder_ckpt: str       = "",
    lazy_morph_ckpt: str       = "",
    lazy_morph_step: int       = 0,
    output_suffix: str         = "",
    out_dir: str               = "",
) -> None:
    import numpy as np

    out_path = Path(out_dir) if out_dir else HERE
    out_path.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, list[dict]] = {}

    shared = dict(
        n_samples=n_samples, parquet=parquet,
        temperature=temperature, random_seed=random_seed, do_speedup=do_speedup,
    )
    print("Spawning 3 model containers in parallel...")

    sc_path = supercoder_ckpt or EXP1_DEFAULT_CKPT

    calls = [
        eval_model.spawn(
            tag="qwen-base",
            model_path=QWEN_BASE,
            exp_name="",
            is_lazy=False,
            **shared,
        ),
        eval_model.spawn(
            tag="supercoder",
            model_path=sc_path,
            exp_name="",
            is_lazy=False,
            **shared,
        ),
        eval_model.spawn(
            tag="lazy-morph",
            model_path=lazy_morph_ckpt,
            exp_name=LAZY_EXP,
            exp_step=lazy_morph_step,
            is_lazy=True,
            **shared,
        ),
    ]
    print("Waiting for all containers to finish...\n")
    for call in calls:
        tag, results = call.get()
        if results:
            all_results[tag] = results
            print(f"  [{tag}] finished -- {len(results)} samples")
        else:
            print(f"  [{tag}] skipped (no checkpoint found)")

    # Write CSVs
    detail_fields = [
        "idx", "problem_idx", "status", "compiled", "tests_pass", "correctness",
        "raw_speedup", "effective_speedup", "speedup_floor1", "is_copy",
        "response_tokens", "n_inputs", "compile_stderr", "response",
        "morph_called", "morph_success",
    ]
    summary_rows = []
    suffix = output_suffix

    for tag, results in all_results.items():
        csv_path = out_path / f"infer_results_{tag}{suffix}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=detail_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved {len(results)} rows -> {csv_path}")

        n            = len(results)
        sp_vals      = [r["raw_speedup"] for r in results if r["raw_speedup"] is not None]
        eff_vals     = [r["effective_speedup"] for r in results]
        floor1_vals  = [r["speedup_floor1"]    for r in results]
        geo_floor1 = (
            float(np.exp(np.mean(np.log(floor1_vals)))) if floor1_vals else 1.0
        )
        status_counts: dict[str, int] = {}
        for r in results:
            status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

        row = {
            "model":                tag,
            "n_samples":            n,
            "compile_rate":         round(sum(r["compiled"]    for r in results) / n, 3),
            "test_pass_rate":       round(sum(r["tests_pass"]  for r in results) / n, 3),
            "mean_correctness":     round(sum(r["correctness"] for r in results) / n, 3),
            "n_speedup_measured":   len(sp_vals),
            "mean_speedup":         round(float(np.mean(sp_vals)), 4) if sp_vals else "",
            "max_speedup":          round(float(max(sp_vals)),     4) if sp_vals else "",
            "mean_effective_speedup": round(float(np.mean(eff_vals)), 4) if eff_vals else "",
            "geo_mean_speedup_floor1": round(max(1.0, geo_floor1), 4),
            "p25_speedup_floor1":   round(float(np.percentile(floor1_vals, 25)), 4),
            "p50_speedup_floor1":   round(float(np.percentile(floor1_vals, 50)), 4),
            "p75_speedup_floor1":   round(float(np.percentile(floor1_vals, 75)), 4),
            "copy_rate":            round(sum(r["is_copy"] for r in results) / n, 3),
            "status_breakdown":     str(status_counts),
        }
        # Morph stats for lazy-morph
        if any(r.get("morph_called") is not None for r in results):
            n_morph = sum(1 for r in results if r.get("morph_called"))
            n_morph_ok = sum(1 for r in results if r.get("morph_success"))
            row["morph_call_rate"]    = round(n_morph / n, 3)
            row["morph_success_rate"] = round(n_morph_ok / max(n_morph, 1), 3)

        summary_rows.append(row)

    summary_path = out_path / f"infer_summary{suffix}.csv"
    summary_fields = [
        "model", "n_samples", "compile_rate", "test_pass_rate", "mean_correctness",
        "n_speedup_measured", "mean_speedup", "max_speedup", "mean_effective_speedup",
        "geo_mean_speedup_floor1", "p25_speedup_floor1", "p50_speedup_floor1",
        "p75_speedup_floor1", "copy_rate", "status_breakdown",
        "morph_call_rate", "morph_success_rate",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary -> {summary_path}")

    # Pretty-print comparison table
    col_w = 20
    print(f"\n{'='*85}")
    print(f"COMPARISON  (parquet={parquet}, n={n_samples}, temp={temperature}, "
          f"speedup={'on' if do_speedup else 'off'})")
    print(f"{'='*85}")
    hdr = f"{'metric':<26}" + "".join(f"{r['model']:>{col_w}}" for r in summary_rows)
    print(hdr)
    print("-" * len(hdr))
    for key, label in [
        ("compile_rate",            "compile_rate"),
        ("test_pass_rate",          "test_pass_rate"),
        ("mean_correctness",        "mean_correctness"),
        ("mean_speedup",            "mean_speedup (ALL_PASS)"),
        ("max_speedup",             "max_speedup"),
        ("mean_effective_speedup",  "mean_effective_speedup"),
        ("geo_mean_speedup_floor1", "geo_mean (paper)"),
        ("p50_speedup_floor1",      "p50_speedup"),
        ("p75_speedup_floor1",      "p75_speedup"),
        ("copy_rate",               "copy_rate (bad)"),
        ("morph_call_rate",         "morph_call_rate"),
        ("morph_success_rate",      "morph_success_rate"),
    ]:
        print(f"{label:<26}" + "".join(f"{str(r.get(key, '')):>{col_w}}" for r in summary_rows))
    print()
