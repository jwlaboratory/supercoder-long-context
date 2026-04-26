"""Step B — Benchmarking inside Docker (Linux x86-64).

Takes the per-model CSVs from step A and for each sample:
  1. compile the generated assembly with `gcc` (inside Docker)
  2. run every test case (stdin → check stdout)
  3. if ALL tests pass, time the generated vs. unoptimized binary and compute
     speedup = unopt_ms / gen_ms

The model outputs are **GNU ELF x86-64 assembly** and can't be assembled by
macOS `gcc`/`clang`. This script delegates all compilation and execution to
the `supercoder-x86-bench` image (Ubuntu 22.04 + gcc + python3) — the same
one used by `qwen-debug-rl/1-gen-training-data/3-identify-fails.py`.

If you haven't built it yet:

    docker build --platform linux/amd64 \\
        -f old-experiments/experiments/run-their-paper-exactly/docker/x86_64-benchmark.Dockerfile \\
        -t supercoder-x86-bench .

⚠ Running under Docker Desktop on Apple Silicon uses QEMU emulation:
    correctness → reliable
    timings     → unreliable (you're measuring the emulator)
  For trustworthy speedups, run on a native x86-64 Linux host.

Writes a new enriched CSV next to each input:
    infer_<tag>.csv   →   infer_<tag>__benchmarked.csv

Added columns:
    compile_gen_ok         bool
    compile_stderr         str (truncated)
    is_copy                bool   — generated == unoptimized (model cheated)
    n_tests                int
    n_tests_pass           int
    test_pass_rate         float  (n_tests_pass / n_tests)
    all_tests_pass         bool
    unopt_mean_ms          float | "" — only if all_tests_pass
    gen_mean_ms            float | ""
    speedup                float | ""   (unopt_mean_ms / gen_mean_ms)

Usage:
    cd qwen-long-context/1-experiment-supercoder-long/eval
    uv run python b-benchmark-local.py                              # all CSVs
    uv run python b-benchmark-local.py --no-speedup                 # correctness only
    uv run python b-benchmark-local.py --max-rows 20                # smoke test
    uv run python b-benchmark-local.py --csv results/infer_qwen-base.csv
    uv run python b-benchmark-local.py --docker-image my-custom:tag
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

HERE                 = Path(__file__).resolve().parent
RESULTS_DIR          = HERE / "results"
DOCKER_IMAGE_DEFAULT = "supercoder-x86-bench"
BATCH_SIZE_DEFAULT   = 100
MAX_TESTS_DEFAULT    = 10       # correctness: run up to N tests per sample
MAX_TESTS_BENCH_DEF  = 3        # timing: run hyperfine on up to N inputs
RUNS_DEFAULT         = 5
WARMUP_DEFAULT       = 2
TIMEOUT_DEFAULT      = 10.0     # per-run timeout (seconds)


# ── Runner script that executes inside Docker (stdlib only) ──────────────────
#
# Uses `time.perf_counter_ns()` rather than `hyperfine` so the image only needs
# gcc + python3 (matches `supercoder-x86-bench`). Includes a warmup loop and N
# timed runs; process fork/exec overhead is identical for both binaries so the
# ratio is still meaningful.
_DOCKER_RUNNER = r"""#!/usr/bin/env python3
import json, os, subprocess, sys, time
from pathlib import Path

work           = Path(sys.argv[1])
config         = json.loads((work / "config.json").read_text())
DO_SPEEDUP     = bool(config["do_speedup"])
RUNS           = int(config["runs"])
WARMUP         = int(config["warmup"])
TIMEOUT_S      = float(config["timeout_s"])
MAX_TESTS      = int(config["max_tests"])
MAX_TESTS_BEN  = int(config["max_tests_bench"])


def compile_asm(asm_path, bin_path):
    try:
        r = subprocess.run(
            ["gcc", str(asm_path), "-o", str(bin_path), "-lm"],
            capture_output=True, text=True, timeout=45,
        )
    except subprocess.TimeoutExpired:
        return False, "compile timeout (45s)"
    if r.returncode != 0 or not bin_path.exists():
        return False, ("compile error:\n" + (r.stderr or ""))[:400]
    return True, ""


def run_once(bin_path, stdin_text):
    try:
        r = subprocess.run(
            [str(bin_path)], input=stdin_text,
            capture_output=True, text=True, timeout=TIMEOUT_S,
        )
        return r.returncode == 0, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return False, "", "timeout"
    except Exception as e:
        return False, "", f"exception: {e}"[:300]


def time_runs(bin_path, stdin_text, runs, warmup):
    for _ in range(warmup):
        try:
            subprocess.run(
                [str(bin_path)], input=stdin_text,
                capture_output=True, text=True, timeout=TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return None
    samples_ms = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        try:
            subprocess.run(
                [str(bin_path)], input=stdin_text,
                capture_output=True, text=True, timeout=TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return None
        samples_ms.append((time.perf_counter_ns() - t0) / 1e6)
    return sum(samples_ms) / len(samples_ms)


results = {}
row_dirs = sorted(p for p in work.glob("row_*") if p.is_dir())
n_total  = len(row_dirs)

for i, row_dir in enumerate(row_dirs, 1):
    idx     = row_dir.name[4:]
    gen_s   = row_dir / "gen.s"
    unopt_s = row_dir / "unopt.s"
    tests   = json.loads((row_dir / "tests.json").read_text())
    tests   = tests[:MAX_TESTS]

    res = {
        "compile_gen_ok":  False,
        "compile_stderr":  "",
        "n_tests":         len(tests),
        "n_tests_pass":    0,
        "all_tests_pass":  False,
        "unopt_mean_ms":   None,
        "gen_mean_ms":     None,
        "speedup":         None,
    }

    # -- compile generated
    if not gen_s.exists() or gen_s.stat().st_size == 0:
        res["compile_stderr"] = "empty_response"
        results[idx] = res
        print(f"[{i:4d}/{n_total}] idx={idx} compile=N (empty)", flush=True)
        continue

    gen_bin = row_dir / "gen.bin"
    ok, stderr = compile_asm(gen_s, gen_bin)
    res["compile_gen_ok"] = ok
    res["compile_stderr"] = stderr
    if not ok:
        results[idx] = res
        print(f"[{i:4d}/{n_total}] idx={idx} compile=N", flush=True)
        continue

    # -- correctness
    n_pass = 0
    for t in tests:
        rc_ok, got, _ = run_once(gen_bin, t.get("input", ""))
        if rc_ok and got.rstrip() == (t.get("output", "") or "").rstrip():
            n_pass += 1
    res["n_tests_pass"]   = n_pass
    all_pass              = (len(tests) > 0 and n_pass == len(tests))
    res["all_tests_pass"] = all_pass

    # -- speedup
    if DO_SPEEDUP and all_pass and unopt_s.exists():
        unopt_bin = row_dir / "unopt.bin"
        ok_u, _   = compile_asm(unopt_s, unopt_bin)
        if ok_u:
            u_means, g_means = [], []
            for t in tests[:MAX_TESTS_BEN]:
                stdin_text = t.get("input", "")
                u_ms = time_runs(unopt_bin, stdin_text, RUNS, WARMUP)
                g_ms = time_runs(gen_bin,   stdin_text, RUNS, WARMUP)
                if u_ms is not None and g_ms is not None and g_ms > 0:
                    u_means.append(u_ms)
                    g_means.append(g_ms)
            if u_means:
                u_avg = sum(u_means) / len(u_means)
                g_avg = sum(g_means) / len(g_means)
                res["unopt_mean_ms"] = round(u_avg, 3)
                res["gen_mean_ms"]   = round(g_avg, 3)
                res["speedup"]       = round(u_avg / g_avg, 4) if g_avg > 0 else None

    sp = res["speedup"]
    sp_txt = f" speedup={sp}" if sp is not None else ""
    print(
        f"[{i:4d}/{n_total}] idx={idx} compile=Y "
        f"pass={n_pass}/{len(tests)}{sp_txt}",
        flush=True,
    )
    results[idx] = res

(work / "results.json").write_text(json.dumps(results))
"""


# ── Host-side helpers ────────────────────────────────────────────────────────

def _check_docker() -> None:
    if shutil.which("docker") is None:
        raise SystemExit("docker not found — install Docker Desktop first.")
    r = subprocess.run(["docker", "info"], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(
            "docker daemon not reachable. Start Docker Desktop and retry.\n"
            f"stderr: {r.stderr.strip()}"
        )


def _ensure_image(image: str) -> None:
    r = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise SystemExit(
            f"Docker image not found: {image!r}.\n"
            f"Build it with:\n"
            f"  docker build --platform linux/amd64 \\\n"
            f"      -f old-experiments/experiments/run-their-paper-exactly/docker/"
            f"x86_64-benchmark.Dockerfile \\\n"
            f"      -t {image} .\n"
            f"or pass --docker-image <tag> to use a different one."
        )


def _run_batch(
    batch_rows:      list[dict],
    docker_image:    str,
    do_speedup:      bool,
    runs:            int,
    warmup:          int,
    timeout_s:       float,
    max_tests:       int,
    max_tests_bench: int,
) -> dict[str, dict]:
    """Write batch into a temp dir, docker run the runner, parse results.json."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)

        for row in batch_rows:
            idx = row["_bench_idx"]
            row_dir = work / f"row_{idx}"
            row_dir.mkdir()
            (row_dir / "gen.s").write_text(row.get("generated_assembly", "") or "",
                                           encoding="utf-8")
            (row_dir / "unopt.s").write_text(row.get("unoptimized_assembly", "") or "",
                                             encoding="utf-8")
            try:
                tests = json.loads(row.get("test_cases") or "[]")
            except Exception:
                tests = []
            (row_dir / "tests.json").write_text(
                json.dumps(tests, ensure_ascii=False), encoding="utf-8",
            )

        (work / "config.json").write_text(json.dumps({
            "do_speedup":      do_speedup,
            "runs":            runs,
            "warmup":          warmup,
            "timeout_s":       timeout_s,
            "max_tests":       max_tests,
            "max_tests_bench": max_tests_bench,
        }))
        (work / "runner.py").write_text(_DOCKER_RUNNER, encoding="utf-8")

        proc = subprocess.run([
            "docker", "run", "--rm", "--platform", "linux/amd64",
            "-v", f"{work}:/work",
            docker_image,
            "python3", "/work/runner.py", "/work",
        ])

        results_file = work / "results.json"
        if results_file.exists():
            return json.loads(results_file.read_text(encoding="utf-8"))

        print(
            f"  WARNING: docker exit={proc.returncode}, no results.json. "
            f"Marking batch rows as unknown compile failure.",
            file=sys.stderr,
        )
        return {
            str(row["_bench_idx"]): {
                "compile_gen_ok":  False,
                "compile_stderr":  f"docker exit {proc.returncode}",
                "n_tests":         0,
                "n_tests_pass":    0,
                "all_tests_pass":  False,
                "unopt_mean_ms":   None,
                "gen_mean_ms":     None,
                "speedup":         None,
            }
            for row in batch_rows
        }


# ── Per-CSV orchestration ────────────────────────────────────────────────────

EXTRA_COLS = [
    "compile_gen_ok", "compile_stderr", "is_copy",
    "n_tests", "n_tests_pass", "test_pass_rate", "all_tests_pass",
    "unopt_mean_ms", "gen_mean_ms", "speedup",
]


def _process_csv(
    in_path:    Path,
    out_path:   Path,
    args:       argparse.Namespace,
) -> None:
    csv.field_size_limit(sys.maxsize)
    with in_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.max_rows:
        rows = rows[: args.max_rows]
    for i, row in enumerate(rows):
        row["_bench_idx"] = i
    print(f"\n=== {in_path.name} — {len(rows)} rows ===")

    # Tag is_copy on host before handing to docker
    for row in rows:
        gen   = (row.get("generated_assembly", "") or "").strip()
        unopt = (row.get("unoptimized_assembly", "") or "").strip()
        row["is_copy"] = (gen != "" and gen == unopt)

    # Batch through Docker
    results: dict[str, dict] = {}
    n_batches = (len(rows) + args.batch_size - 1) // args.batch_size
    t0 = time.time()
    for bi in range(n_batches):
        start = bi * args.batch_size
        end   = min(start + args.batch_size, len(rows))
        batch = rows[start:end]
        print(
            f"--- batch {bi+1}/{n_batches} (rows {start}-{end-1}, n={len(batch)}) "
            f"[{args.docker_image}] ---"
        )
        results.update(_run_batch(
            batch,
            docker_image=args.docker_image,
            do_speedup=not args.no_speedup,
            runs=args.runs,
            warmup=args.warmup,
            timeout_s=args.timeout_s,
            max_tests=args.max_tests,
            max_tests_bench=args.max_tests_bench,
        ))

    # Merge results back into rows
    out_fields = [k for k in rows[0].keys() if k != "_bench_idx"] + [
        c for c in EXTRA_COLS if c not in rows[0]
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            res: dict[str, Any] = results.get(str(row["_bench_idx"]), {})
            row.pop("_bench_idx", None)
            row["compile_gen_ok"] = bool(res.get("compile_gen_ok"))
            row["compile_stderr"] = res.get("compile_stderr", "")
            n_tests     = int(res.get("n_tests", 0))
            n_pass      = int(res.get("n_tests_pass", 0))
            row["n_tests"]        = n_tests
            row["n_tests_pass"]   = n_pass
            row["test_pass_rate"] = round(n_pass / n_tests, 4) if n_tests else 0.0
            row["all_tests_pass"] = bool(res.get("all_tests_pass"))
            row["unopt_mean_ms"]  = res.get("unopt_mean_ms") or ""
            row["gen_mean_ms"]    = res.get("gen_mean_ms")   or ""
            row["speedup"]        = res.get("speedup")       or ""
            w.writerow(row)
    print(f"wrote {len(rows)} rows → {out_path}  ({time.time()-t0:.1f}s)")


# ── Entrypoint ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv",   type=Path, default=None,
                   help="Specific infer_*.csv to process (default: all in ./results/).")
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--docker-image", default=DOCKER_IMAGE_DEFAULT,
                   help=f"Linux x86-64 image with gcc+python3 (default: {DOCKER_IMAGE_DEFAULT}).")
    p.add_argument("--no-speedup", action="store_true",
                   help="Skip timing; only measure correctness.")
    p.add_argument("--runs",   type=int, default=RUNS_DEFAULT,
                   help=f"Timed runs per test input (default: {RUNS_DEFAULT}).")
    p.add_argument("--warmup", type=int, default=WARMUP_DEFAULT,
                   help=f"Warmup runs before timed runs (default: {WARMUP_DEFAULT}).")
    p.add_argument("--timeout-s", type=float, default=TIMEOUT_DEFAULT,
                   help=f"Per-invocation timeout in seconds (default: {TIMEOUT_DEFAULT}).")
    p.add_argument("--max-tests", type=int, default=MAX_TESTS_DEFAULT,
                   help=f"Max test cases run for correctness (default: {MAX_TESTS_DEFAULT}).")
    p.add_argument("--max-tests-bench", type=int, default=MAX_TESTS_BENCH_DEF,
                   help=f"Max test inputs timed for speedup (default: {MAX_TESTS_BENCH_DEF}).")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT,
                   help=f"Rows per docker invocation (default: {BATCH_SIZE_DEFAULT}).")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Cap rows per CSV (for smoke tests).")
    args = p.parse_args()

    _check_docker()
    _ensure_image(args.docker_image)

    if platform.machine().lower() == "arm64" and not args.no_speedup:
        print(
            "\n  ⚠  Apple Silicon detected. Docker runs linux/amd64 under QEMU:\n"
            "     correctness → reliable, timings → unreliable (you're measuring\n"
            "     the emulator). Add --no-speedup for pure correctness, or run on\n"
            "     a native x86-64 Linux host for real speedup numbers.\n",
            file=sys.stderr,
        )

    if args.csv:
        csvs = [args.csv]
    else:
        csvs = sorted(args.results_dir.glob("infer_*.csv"))
        csvs = [c for c in csvs if not c.name.endswith("__benchmarked.csv")]
    if not csvs:
        raise SystemExit(f"No infer_*.csv found in {args.results_dir}")

    for in_path in csvs:
        out_path = in_path.with_name(in_path.stem + "__benchmarked.csv")
        _process_csv(in_path, out_path, args)


if __name__ == "__main__":
    main()
