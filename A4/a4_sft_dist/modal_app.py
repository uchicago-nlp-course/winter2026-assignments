import modal

app = modal.App("a4-sft")

# ---------------------------------------------------------------------------
# Volumes (persistent across runs)
# ---------------------------------------------------------------------------
hf_cache_vol = modal.Volume.from_name("a4-hf-cache", create_if_missing=True)
output_vol = modal.Volume.from_name("a4-output", create_if_missing=True)
eval_results_vol = modal.Volume.from_name("a4-eval-results", create_if_missing=True)

# ---------------------------------------------------------------------------
# Image: mount entire a4_sft_solutions directory + install dependencies
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers==4.57.6",
        "accelerate",
        "datasets",
        "wandb",
        "tqdm",
        "scikit-learn",
        "nltk",
        "pandas",
        "numpy",
        "sentencepiece",
        "protobuf",
        gpu="A100",
    )
    .add_local_dir(
        ".",
        remote_path="/root/a4_sft_dist",
        ignore=["out/", "eval_results/", "wandb/"],
    )
)

# ---------------------------------------------------------------------------
# Shared resource configuration
# ---------------------------------------------------------------------------
SHARED_KWARGS = dict(
    image=image,
    gpu="A100-80GB",
    cpu=8.0,
    memory=65536,          # 64 GB in MiB
    timeout=3 * 3600,      # 3-hour max (training can be long)
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/a4_sft_dist/out": output_vol,
        "/root/a4_sft_dist/eval_results": eval_results_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),        # WANDB_API_KEY
        modal.Secret.from_name("huggingface-secret"),   # HF_TOKEN
    ],
)

# Sed pattern: strip hardcoded cache-dir and CUDA exports from shell scripts.
# With these removed, Python code falls back to HF default cache at
# ~/.cache/huggingface (volume-backed), and Modal manages CUDA devices.
_SED_STRIP = r"/^export (MODEL|DATA)_CACHE_DIR=/d; /^export CUDA_VISIBLE_DEVICES=/d"


# ---------------------------------------------------------------------------
# Training function (remote GPU): runs scripts/run_sft_train.sh
# ---------------------------------------------------------------------------
@app.function(**SHARED_KWARGS)
def run_train_scripts():
    import subprocess

    cmd = f"sed -E '{_SED_STRIP}' scripts/run_sft_train.sh | bash"
    subprocess.run(["bash", "-c", cmd], cwd="/root/a4_sft_dist", check=True)

    output_vol.commit()
    hf_cache_vol.commit()


# ---------------------------------------------------------------------------
# Evaluation function (remote GPU): runs scripts/run_sft_eval.sh
# ---------------------------------------------------------------------------
@app.function(**SHARED_KWARGS)
def run_eval_scripts():
    import subprocess

    cmd = f"sed -E '{_SED_STRIP}' scripts/run_sft_eval.sh | bash"
    subprocess.run(["bash", "-c", cmd], cwd="/root/a4_sft_dist", check=True)

    eval_results_vol.commit()
    hf_cache_vol.commit()


# ---------------------------------------------------------------------------
# Local entrypoints: orchestrate remote functions from the CLI
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def launch_train():
    run_train_scripts.remote()


@app.local_entrypoint()
def launch_eval():
    run_eval_scripts.remote()


@app.local_entrypoint()
def launch_all():
    run_train_scripts.remote()
    run_eval_scripts.remote()
