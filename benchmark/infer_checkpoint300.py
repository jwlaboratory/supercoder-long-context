"""Benchmark the train2-lazy-supercoder global_step_300 checkpoint.

This is a thin wrapper around ``infer.py`` so the normal benchmark stays
available while this file always targets the last healthy lazy checkpoint.

Usage
-----
    cd qwen-long-context/benchmark
    modal run merge_checkpoint.py --exp train2-lazy-supercoder --step 300
    modal run infer_checkpoint300.py
    modal run infer_checkpoint300.py --do-speedup

Outputs are written to ``checkpoint300_results/`` with ``_300`` filenames,
for example ``infer_summary_300.csv``.
"""
from __future__ import annotations

from pathlib import Path

from infer import HERE, app, main as run_benchmark


@app.local_entrypoint(name="checkpoint300")
def checkpoint300(
    n_samples: int = 200,
    parquet: str = "sc_val",
    temperature: float = 0.0,
    random_seed: int = 42,
    do_speedup: bool = False,
    supercoder_ckpt: str = "",
    lazy_morph_ckpt: str = "",
    out_dir: str = "",
) -> None:
    step300_out_dir = out_dir or str(Path(HERE) / "checkpoint300_results")
    run_benchmark(
        n_samples=n_samples,
        parquet=parquet,
        temperature=temperature,
        random_seed=random_seed,
        do_speedup=do_speedup,
        supercoder_ckpt=supercoder_ckpt,
        lazy_morph_ckpt=lazy_morph_ckpt,
        lazy_morph_step=300,
        output_suffix="_300",
        out_dir=step300_out_dir,
    )
