import modal

app = modal.App("a4-sampling")

# ---------------------------------------------------------------------------
# Volumes (persistent across runs)
# ---------------------------------------------------------------------------
hf_cache_vol = modal.Volume.from_name("a4-hf-cache", create_if_missing=True)
# This volume contains the SFT checkpoints from Part 1
output_vol = modal.Volume.from_name("a4-output", create_if_missing=True)
sampling_results_vol = modal.Volume.from_name("a4-sampling-results", create_if_missing=True)

# ---------------------------------------------------------------------------
# Image: use the same configuration as Part 1 to benefit from image caching
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
        remote_path="/root/a4_sampling",
        ignore=["results/"],
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
    timeout=3 * 3600,      # 3-hour max
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/a4_sampling/sft_out": output_vol,
        "/root/a4_sampling/results": sampling_results_vol,
    },
    secrets=[
        # Use the same HF secret as in Part 1
        modal.Secret.from_name("huggingface-secret"),
    ],
)

# ---------------------------------------------------------------------------
# Sampling function (remote GPU): runs scripts/generate_samples.sh
# ---------------------------------------------------------------------------
@app.function(**SHARED_KWARGS)
def run_sampling_scripts(model: str = "1b"):
    import subprocess
    import os

    # Run the generation script with the model choice (1b, 4b, all)
    results_dir = "results/sampling"
    cmd = f"bash scripts/generate_samples.sh '{model}' '{results_dir}'"
    subprocess.run(["bash", "-c", cmd], cwd="/root/a4_sampling", check=True)

    sampling_results_vol.commit()
    hf_cache_vol.commit()


# ---------------------------------------------------------------------------
# Local entrypoints: orchestrate remote functions from the CLI
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def launch_sampling(model: str = "1b"):
    """
    Start the sampling process on Modal.
    Args:
        model: '1b', '4b', or 'all' (defaults to '1b')
    """
    run_sampling_scripts.remote(model)
