"""Convert a verl FSDP actor checkpoint -> HuggingFace model directory.

The merged HF model is saved back into the same Modal volume so inference can load it.

Usage:
    modal run merge_checkpoint.py                                    # latest step
    modal run merge_checkpoint.py --step 200
    modal run merge_checkpoint.py --exp train2-lazy-supercoder --step 200
"""
from __future__ import annotations
from pathlib import Path
import modal

MINUTES       = 60
HERE          = Path(__file__).resolve().parent
VERL_DIR      = (HERE / "../training/verl").resolve()
BASE_MODEL    = "Qwen/Qwen2.5-Coder-7B-Instruct"

app             = modal.App("merge-lazy-supercoder-checkpoint")
hf_secret       = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])
checkpoints_vol = modal.Volume.from_name("debug-rl-checkpoints", create_if_missing=False)
hf_cache_vol    = modal.Volume.from_name("huggingface-cache",    create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.40,<5",
        "safetensors",
        "numpy",
        "huggingface_hub",
    )
    .add_local_file(str(VERL_DIR / "scripts/model_merger.py"), "/model_merger.py")
)


@app.function(
    image=image,
    cpu=8,
    memory=65536,
    timeout=60 * MINUTES,
    secrets=[hf_secret],
    volumes={
        "/checkpoints":             checkpoints_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
)
def merge(exp: str = "train2-lazy-supercoder", step: int = 398) -> str:
    import os, re, sys, subprocess

    root = f"/checkpoints/{exp}"
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Experiment dir not found: {root}")

    # Auto-detect latest step if not specified
    if step == 0:
        step_dirs = sorted(
            [int(m.group(1)) for name in os.listdir(root)
             if (m := re.match(r"global_step_(\d+)$", name))],
            reverse=True,
        )
        if not step_dirs:
            raise FileNotFoundError(f"No global_step_N dirs in {root}")
        step = step_dirs[0]
        print(f"Auto-detected latest step: {step}")

    step_dir  = f"{root}/global_step_{step}"
    actor_dir = f"{step_dir}/actor"
    hf_dir    = f"{step_dir}/hf_model"

    if not os.path.isdir(actor_dir):
        raise FileNotFoundError(
            f"Actor checkpoint not found at {actor_dir}\n"
            f"Contents of {step_dir}: {os.listdir(step_dir) if os.path.isdir(step_dir) else 'MISSING'}"
        )

    pt_files = [f for f in os.listdir(actor_dir) if f.endswith(".pt")]
    print(f"Found {len(pt_files)} shard file(s) in {actor_dir}:")
    for f in sorted(pt_files):
        size_gb = os.path.getsize(os.path.join(actor_dir, f)) / 1e9
        print(f"  {f}  ({size_gb:.1f} GB)")

    print(f"\nMerging -> {hf_dir}")
    subprocess.run(
        [
            sys.executable, "/model_merger.py",
            "--backend",       "fsdp",
            "--hf_model_path", BASE_MODEL,
            "--local_dir",     actor_dir,
            "--target_dir",    hf_dir,
        ],
        check=True,
    )

    merged_files = os.listdir(hf_dir)
    print(f"\nMerged model files: {merged_files}")

    checkpoints_vol.commit()
    print(f"\nCommitted volume. HF model at: {hf_dir}")
    return hf_dir


@app.local_entrypoint()
def main(exp: str = "train2-lazy-supercoder", step: int = 398) -> None:
    hf_path = merge.remote(exp=exp, step=step)
    print(f"\n{'='*60}")
    print(f"Merged model saved to Modal volume at:")
    print(f"  {hf_path}")
    print(f"{'='*60}")
