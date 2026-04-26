# Long-context bucket evaluation

Compare `Qwen/Qwen2.5-Coder-7B-Instruct` (base) vs your trained supercoder
(defaults to `/checkpoints/exp1-train-supercoder/global_step_420/hf_model` on
the `debug-rl-checkpoints` Modal volume — override with `--supercoder-path`)
on the 4 C-length buckets built in `../data/` — 200 random samples per bucket
by default.

## Pipeline

| Step | Script | Where it runs | What it does |
|---|---|---|---|
| A | `a-inference.py` | Modal (1× H100 per model, parallel) | Compile C → unopt asm, build supercoder-style prompt, vLLM generate optimized asm. Writes one CSV per model + a manifest. |
| B | `b-benchmark-local.py` | Your PC | Compile generated asm with `gcc`, run every test case (stdin→stdout), hyperfine-benchmark passing samples vs unopt. |
| C | `c-visualize.py` | Your PC | Plot compile / pass / speedup per bucket and write a summary CSV. |

Splitting benchmarking onto the local machine gives you a consistent CPU timing
environment (no noisy shared-GPU node) and avoids paying for Modal seconds you
don't need.

## Step A — Inference

```bash
cd qwen-long-context/1-experiment-supercoder-long/eval
modal run a-inference.py                                         # 200/bucket, both models
modal run a-inference.py --n-per-bucket 50 --models qwen-base    # quick smoke
modal run a-inference.py --models supercoder                     # re-run only supercoder
modal run a-inference.py --supercoder-path /checkpoints/exp1-train-supercoder/global_step_420/hf_model
modal run a-inference.py --supercoder-path random1123anonymized/supercoder   # HF hub id (needs access)
modal run a-inference.py --temperature 0 --max-tokens 6000       # deterministic, longer outputs
```

Output:
```
results/
  infer_qwen-base.csv
  infer_supercoder.csv
  manifest.json
```

## Step B — Benchmark

The generated assembly is **GNU ELF x86-64** (`.section ...,@progbits`,
`.type main, @function`, `.cfi_*`, etc.) which macOS `gcc`/`clang` can't
assemble. `b-benchmark-local.py` delegates compile + test + timing to the
**`supercoder-x86-bench`** Docker image (Ubuntu 22.04 + gcc + python3) —
same one used by `qwen-debug-rl/1-gen-training-data/3-identify-fails.py`.

**One-time image build (skip if you already have it):**

```bash
docker build --platform linux/amd64 \
  -f old-experiments/experiments/run-their-paper-exactly/docker/x86_64-benchmark.Dockerfile \
  -t supercoder-x86-bench .
```

**Run:**

```bash
cd qwen-long-context/1-experiment-supercoder-long/eval
uv run python b-benchmark-local.py                       # all CSVs in results/
uv run python b-benchmark-local.py --no-speedup          # correctness only (fast)
uv run python b-benchmark-local.py --max-rows 20         # smoke test
uv run python b-benchmark-local.py --csv results/infer_qwen-base.csv
uv run python b-benchmark-local.py --docker-image other-tag:v2
```

The script shells out to `docker run ... supercoder-x86-bench python3 /work/runner.py`
per batch (default 100 rows/invocation). Timing is done with `time.perf_counter_ns()`
inside the container — no `hyperfine` dependency.

⚠ On Apple Silicon, Docker runs linux/amd64 under QEMU: **correctness is
reliable, but speedups are not** (you're timing the emulator, not the CPU).
Add `--no-speedup` for a fast correctness pass, or run on a native x86-64
Linux host for trustworthy speedup numbers.

Each input `infer_<tag>.csv` gets a sibling `infer_<tag>__benchmarked.csv` with
added columns: `compile_gen_ok`, `n_tests_pass`, `all_tests_pass`,
`unopt_mean_ms`, `gen_mean_ms`, `speedup`, etc.

## Step C — Visualize

```bash
cd qwen-long-context/1-experiment-supercoder-long/eval
uv run python c-visualize.py
```

Produces:
```
results/summary_by_bucket.csv
results/figures/compile_rate_by_bucket.png
results/figures/pass_rate_by_bucket.png
results/figures/mean_speedup_by_bucket.png
results/figures/speedup_distribution.png
```

## Notes

- Both models see **exactly the same sampled rows** (seed=42, `n_per_bucket=200`).
  The manifest records `sampled_indices` so step B/C can verify.
- The prompt is identical to supercoder training data (`Given the following C
  code and assembly code, your task is to generate highly optimized x86-64
  assembly code…` + ` ```c… ``` ` + ` ```assembly… ``` `). C is compiled to
  unoptimized asm inside the Modal container with `gcc -S -O0` for a
  deterministic template across models.
- `b-benchmark-local.py` defaults to `--hyperfine-runs 5 --warmup 2
  --max-tests-bench 3`. Bump `--hyperfine-runs` for stable timings on small
  inputs at the cost of runtime.
- `is_copy` flags samples where the model just regurgitated the unoptimized
  assembly verbatim — treat those as a correctness win but a 1.0× speedup.
