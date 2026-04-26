"""step b:

we use morph to fast apply the edits into the code

"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import modal

MINUTES   = 60
HERE      = Path(__file__).resolve().parent
DATA_DIR  = (HERE / "../data/codenet_balanced_hf").resolve()

QWEN_BASE               = "Qwen/Qwen2.5-Coder-7B-Instruct"
# Default supercoder = the exp1-train-supercoder step-420 HF copy on the Modal
# volume. This is the actual model the training runs loaded (the Hub id
# `random1123anonymized/supercoder` is gated and your HF token may not have
# access). Pass --supercoder-path to override.
DEFAULT_SUPERCODER_PATH = "/checkpoints/exp1-train-supercoder/global_step_420/hf_model"
BUCKET_ORDER            = ["short", "medium_short", "medium_long", "long"]

PROMPT_TEMPLATE = (
    "Given the following C code and assembly code, your task is to generate "
    "highly optimized x86-64 assembly code.\n"
    "C Code:\n\n"
    "```c\n{c_code}\n```\n\n"
    "Assembly Code:\n\n"
    "```assembly\n{unopt_asm}\n```\n\n"
    "Only output the optimized assembly code. Do not include any other text. "
    "Do not write any comments in the assembly code. Wrap the assembly code "
    "in ```assembly``` tags."
    
    "Use \"// ... existing code ...\" to represent unchanged code blocks. Include just enough surrounding context to locate each edit precisely."
    
    "Example format:\n"
    "// ... existing code ...\n"
    "FIRST_EDIT\n"
    "// ... existing code ...\n"
    "SECOND_EDIT\n"
    "// ... existing code ...\n"
    "\n"

    "Rules:\n"
    "- ALWAYS use \"// ... existing code ...\" for unchanged sections (omitting this marker will cause deletions)\n"
    "- Include minimal context around edits only when needed for disambiguation\n"
    "- Preserve exact indentation\n"
    "- For deletions: show context before and after, omit the deleted lines\n"
    "- Batch multiple edits to the same file in one call\n"

    "\nOptimized Assembly Code:\n"
)

app = modal.App("long-context-inference")

hf_secret       = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])
hf_cache_vol    = modal.Volume.from_name("huggingface-cache",   create_if_missing=True)
vllm_cache_vol  = modal.Volume.from_name("vllm-cache",          create_if_missing=True)
checkpoints_vol = modal.Volume.from_name("debug-rl-checkpoints", create_if_missing=False)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("gcc")
    .run_commands(
        "pip install torch==2.6.0 torchaudio==2.6.0 torchdata==0.11.0 torchvision==0.21.0"
        " 'ray[default]' psutil cachetools numpy pandas pyarrow",
        "pip install wheel && pip install flash-attn==2.7.4.post1 --no-build-isolation",
        "pip install 'vllm==0.7.3' 'transformers>=4.40,<5' datasets",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_dir(str(DATA_DIR), "/data/codenet_balanced_hf", copy=True)
)


def _extract_assembly(raw: str) -> str:
    """Strip surrounding text / grab the last ```assembly ...``` block."""
    text = raw
    if "```assembly" in text:
        text = text[text.rfind("```assembly") + len("```assembly"):]
    if "```" in text:
        text = text[: text.rfind("```")]
    return text.strip()


@app.function(
    image=image, gpu="h100:1", timeout=4 * 60 * MINUTES,
    secrets=[hf_secret],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm":        vllm_cache_vol,
        "/checkpoints":             checkpoints_vol,
    },
)
def run_model(
    tag: str,
    model_path: str,
    samples: list[dict],
    temperature: float = 0.2,
    max_model_len: int = 16384,
    max_tokens: int = 4096,
) -> tuple[str, list[dict]]:
    """Run inference for one model on the shared sample set.

    `samples` is a list of {bucket, problem_id, submission_id, c_code, ...}.
    Compiles c_code → unopt asm inside the container so gcc version is
    identical for every model.
    """
    import os, subprocess, tempfile, time
    import vllm
    from transformers import AutoTokenizer

    print(f"[{tag}] model={model_path}  n_samples={len(samples)}")
    t_start = time.time()

    # Sanity-check model_path: if it looks like a local Modal volume path but
    # isn't there, fail fast with a useful message (rather than letting HF Hub
    # try to interpret it as a repo id).
    if model_path.startswith("/") and not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"[{tag}] model_path is a local path but does not exist: {model_path}\n"
            f"Check `modal volume ls debug-rl-checkpoints` or pass a different "
            f"--supercoder-path / --qwen-base-path."
        )

    hf_token = os.environ.get("HF_TOKEN") or None

    # Build prompts — compile C → unopt asm per sample
    prompts: list[str] = []
    meta: list[dict]   = []
    for s in samples:
        with tempfile.TemporaryDirectory() as d:
            c_file = os.path.join(d, "x.c")
            s_file = os.path.join(d, "x.s")
            with open(c_file, "w") as f:
                f.write(s["c_code"])
            r = subprocess.run(
                ["gcc", "-S", "-O0", "-o", s_file, c_file],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode != 0 or not os.path.exists(s_file):
                s["unopt_asm"]        = ""
                s["compile_unopt_ok"] = False
                meta.append(s)
                prompts.append("")
                continue
            with open(s_file, "r") as f:
                s["unopt_asm"] = f.read()
            s["compile_unopt_ok"] = True

        meta.append(s)
        prompts.append(PROMPT_TEMPLATE.format(c_code=s["c_code"], unopt_asm=s["unopt_asm"]))

    n_ok = sum(1 for s in meta if s["compile_unopt_ok"])
    print(f"[{tag}] compiled unopt OK: {n_ok}/{len(samples)}")

    # Prepare generation
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, token=hf_token,
    )
    chat_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        ) if p else ""
        for p in prompts
    ]

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

    # Only generate for prompts that compiled; skip empty ones
    gen_idx = [i for i, p in enumerate(chat_prompts) if p]
    gen_prompts = [chat_prompts[i] for i in gen_idx]
    print(f"[{tag}] generating {len(gen_prompts)} / {len(chat_prompts)}")
    t_gen = time.time()
    outputs = llm.generate(gen_prompts, sp)
    print(f"[{tag}] generate done in {time.time()-t_gen:.1f}s")

    # Stitch responses back into meta positions
    out_by_pos: dict[int, tuple[str, int]] = {}
    for out, i in zip(outputs, gen_idx):
        out_by_pos[i] = (out.outputs[0].text, len(out.outputs[0].token_ids))

    rows: list[dict] = []
    for i, s in enumerate(meta):
        raw, n_tok = out_by_pos.get(i, ("", 0))
        rows.append({
            "bucket":              s["bucket"],
            "bucket_idx":          s["bucket_idx"],
            "problem_id":          s["problem_id"],
            "submission_id":       s["submission_id"],
            "c_loc":               s["c_loc"],
            "num_test_cases":      s["num_test_cases"],
            "in_supercoder_train": s.get("in_supercoder_train", False),
            "in_supercoder_val":   s.get("in_supercoder_val",   False),
            "c_code":              s["c_code"],
            "unoptimized_assembly": s["unopt_asm"],
            "compile_unopt_ok":    s["compile_unopt_ok"],
            "raw_response":        raw,
            "generated_assembly":  _extract_assembly(raw) if raw else "",
            "response_tokens":     n_tok,
            "test_cases":          s["test_cases"],   # JSON string (as stored in HF ds)
        })

    print(f"[{tag}] done in {time.time()-t_start:.1f}s")
    return tag, rows


# ── Local entrypoint ─────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    n_per_bucket: int       = 200,
    random_seed: int        = 42,
    temperature: float      = 0.2,
    models: str             = "qwen-base,supercoder",
    max_model_len: int      = 16384,
    max_tokens: int         = 4096,
    out_dir: str            = "",
    qwen_base_path: str     = QWEN_BASE,
    supercoder_path: str    = DEFAULT_SUPERCODER_PATH,
) -> None:
    from datasets import load_from_disk
    import random

    wanted = {m.strip() for m in models.split(",") if m.strip()}
    model_specs: list[tuple[str, str]] = []
    if "qwen-base"  in wanted: model_specs.append(("qwen-base",  qwen_base_path))
    if "supercoder" in wanted: model_specs.append(("supercoder", supercoder_path))
    if not model_specs:
        raise SystemExit(f"--models gave no known tags; got {models!r}")
    for tag, path in model_specs:
        print(f"  model[{tag}] = {path}")

    # ── Sample N per bucket (deterministic) ─────────────────────────────────
    ds = load_from_disk(str(DATA_DIR))
    print(f"Loaded dataset: {len(ds)} rows, columns={ds.column_names}")

    # Note: ds.filter is slow; manual bucket index is fast enough here.
    per_bucket: dict[str, list[int]] = {b: [] for b in BUCKET_ORDER}
    for i, b in enumerate(ds["bucket"]):
        if b in per_bucket:
            per_bucket[b].append(i)

    rng = random.Random(random_seed)
    sampled_idxs: list[int] = []
    for b in BUCKET_ORDER:
        pool = per_bucket[b]
        if len(pool) < n_per_bucket:
            print(f"WARN bucket={b}: only {len(pool)} available (< {n_per_bucket})")
        take = min(n_per_bucket, len(pool))
        sampled_idxs.extend(rng.sample(pool, take))

    print(f"Total samples: {len(sampled_idxs)} "
          f"(expected {n_per_bucket * len(BUCKET_ORDER)})")

    # Build plain python list of samples (small enough to pass over RPC)
    samples: list[dict] = []
    for i in sampled_idxs:
        row = ds[i]
        samples.append({
            "bucket":              row["bucket"],
            "bucket_idx":          row["bucket_idx"],
            "problem_id":          row["problem_id"],
            "submission_id":       row["submission_id"],
            "c_code":              row["c_code"],
            "c_loc":                row["c_loc"],
            "num_test_cases":      row["num_test_cases"],
            "test_cases":          row["test_cases"],   # already JSON string
            "in_supercoder_train": bool(row["in_supercoder_train"]),
            "in_supercoder_val":   bool(row["in_supercoder_val"]),
        })

    # ── Spawn one container per model ──────────────────────────────────────
    out_path = Path(out_dir) if out_dir else (HERE / "results")
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Spawning {len(model_specs)} model container(s)...")
    calls = [
        (tag, run_model.spawn(
            tag=tag, model_path=path, samples=samples,
            temperature=temperature, max_model_len=max_model_len,
            max_tokens=max_tokens,
        ))
        for tag, path in model_specs
    ]

    fields = [
        "bucket", "bucket_idx", "problem_id", "submission_id",
        "c_loc", "num_test_cases", "in_supercoder_train", "in_supercoder_val",
        "c_code", "unoptimized_assembly", "compile_unopt_ok",
        "raw_response", "generated_assembly", "response_tokens",
        "test_cases",
    ]
    csv.field_size_limit(sys.maxsize)

    for tag, call in calls:
        _, rows = call.get()
        csv_path = out_path / f"infer_{tag}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"  [{tag}] wrote {len(rows)} rows → {csv_path}")

    # Write sampling manifest — lets step B/C confirm both models used same set
    manifest = {
        "dataset":         str(DATA_DIR),
        "random_seed":     random_seed,
        "n_per_bucket":    n_per_bucket,
        "temperature":     temperature,
        "max_model_len":   max_model_len,
        "max_tokens":      max_tokens,
        "models":          {t: p for t, p in model_specs},
        "sampled_indices": sampled_idxs,
    }
    with (out_path / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest → {out_path / 'manifest.json'}")
