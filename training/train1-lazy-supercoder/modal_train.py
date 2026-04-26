"""train1-lazy-supercoder: .

(Lazy edit)
Task:   C code + unoptimized assembly → generate faster assembly
Reward: avg speedup if all tests pass, else 0
Data:   random1123anonymized/supercoder (HF)

    modal run modal_train.py
    MODAL_TRAIN_GPU="h100:4" modal run modal_train.py

"""
from __future__ import annotations
import os, subprocess
from pathlib import Path
import modal

MINUTES  = 60
HERE     = Path(__file__).resolve().parent
SHARED   = (HERE / "../shared").resolve()
DATA_DIR = (HERE / "../data").resolve()
VERL_DIR = (HERE / "../verl").resolve()

BASE_MODEL      = "Qwen/Qwen2.5-Coder-7B-Instruct"
EXPERIMENT_NAME = "train1-lazy-supercoder"
TRAIN_FILE      = "/data/sc_train.parquet"
VAL_FILE        = "/data/sc_val.parquet"

app = modal.App(EXPERIMENT_NAME)
hf_secret       = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])
wandb_secret    = modal.Secret.from_name("wandb")
checkpoints_vol = modal.Volume.from_name("debug-rl-checkpoints", create_if_missing=True)
data_vol        = modal.Volume.from_name("debug-rl-data",        create_if_missing=True)
hf_cache_vol    = modal.Volume.from_name("huggingface-cache",    create_if_missing=True)
vllm_cache_vol  = modal.Volume.from_name("vllm-cache",           create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("hyperfine")
    .add_local_dir(str(VERL_DIR), "/verl_src", copy=True)
    .run_commands(
        "pip install torch==2.6.0 torchaudio==2.6.0 torchdata==0.11.0 torchvision==0.21.0"
        " tabulate fire 'ray[default]' psutil cachetools numpy",
        "pip install wheel && pip install flash-attn==2.7.4.post1 --no-build-isolation",
        "pip install -e '/verl_src[vllm]'",
        "pip install 'transformers>=4.40,<5'",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_file(str(SHARED / "reward.py"), "/reward.py")
)

GPU = os.environ.get("MODAL_TRAIN_GPU", "h100:4")


@app.function(
    image=image, gpu=GPU, timeout=24 * 60 * MINUTES,
    secrets=[hf_secret, wandb_secret],
    volumes={
        "/data":                    data_vol,
        "/checkpoints":             checkpoints_vol,
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm":        vllm_cache_vol,
    },
)
def train(model_path: str = BASE_MODEL) -> None:
    subprocess.run(_verl_cmd(model_path), check=True)
    checkpoints_vol.commit()


def _verl_cmd(model_path: str) -> list[str]:
    return [
        "python3", "-m", "verl.trainer.main_ppo",
        "algorithm.adv_estimator=gae",
        f"data.train_files={TRAIN_FILE}",
        f"data.val_files={VAL_FILE}",
        "data.train_batch_size=16",
        "data.max_prompt_length=2000",
        "data.max_response_length=2000",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.ppo_mini_batch_size=16",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.temperature=0.5",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.6",
        "critic.optim.lr=1e-5",
        f"critic.model.path={model_path}",
        "critic.model.use_remove_padding=True",
        "critic.model.enable_gradient_checkpointing=True",
        "critic.ppo_micro_batch_size_per_gpu=2",
        "critic.model.fsdp_config.param_offload=False",
        "critic.model.fsdp_config.optimizer_offload=False",
        "algorithm.use_kl_in_reward=False",
        "trainer.critic_warmup=0",
        "trainer.logger=['console','wandb']",
        "trainer.project_name=qwen-debug-rl",
        f"trainer.experiment_name={EXPERIMENT_NAME}",
        "trainer.n_gpus_per_node=4",
        "trainer.nnodes=1",
        "trainer.save_freq=100",
        "trainer.test_freq=100",
        "trainer.total_epochs=1",
        "trainer.resume_mode=disable",
        f"trainer.default_local_dir=/checkpoints/{EXPERIMENT_NAME}",
        "custom_reward_function.path=/reward.py",
        "custom_reward_function.name=compute_score",
    ]


@app.local_entrypoint()
def main() -> None:
    _ensure_sc_data()
    train.remote()


def _ensure_sc_data():
    train_pq = DATA_DIR / "supercoder_train.parquet"
    val_pq   = DATA_DIR / "supercoder_val.parquet"
    if not train_pq.exists():
        subprocess.run(["uv", "run", "python", "supercoder_to_parquet.py", "--split", "train",
                        "--output-parquet", str(train_pq)], cwd=SHARED, check=True)
    if not val_pq.exists():
        subprocess.run(["uv", "run", "python", "supercoder_to_parquet.py", "--split", "val",
                        "--output-parquet", str(val_pq)], cwd=SHARED, check=True)
    with data_vol.batch_upload(force=True) as u:
        u.put_file(str(train_pq), "sc_train.parquet")
        u.put_file(str(val_pq),   "sc_val.parquet")
