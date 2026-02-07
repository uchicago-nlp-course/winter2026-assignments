#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# One-time Modal setup: authenticate, create secrets, and verify configuration.
# Run this script locally before using `modal run modal_app.py::launch_train` etc.
# ==============================================================================

echo "=========================================="
echo "Modal Setup for A4 SFT"
echo "=========================================="

# --------------------------------------------------------------------------
# 1. Check that the modal CLI is installed
# --------------------------------------------------------------------------
if ! command -v modal &>/dev/null; then
    echo "[ERROR] 'modal' CLI not found. Install it with:"
    echo "  pip install modal"
    exit 1
fi
echo "[OK] modal CLI found: $(modal --version)"

# --------------------------------------------------------------------------
# 2. Authenticate with Modal (skip if already authenticated)
# --------------------------------------------------------------------------
echo ""
echo "[STEP 1] Checking Modal authentication..."
modal_config="${MODAL_CONFIG_PATH:-$HOME/.modal.toml}"
if [[ -f "$modal_config" ]] && grep -q "token_id" "$modal_config" 2>/dev/null; then
    echo "[OK] Existing Modal token found in $modal_config. Skipping modal setup."
else
    echo "  No existing Modal token found. Running 'modal setup'..."
    modal setup
    echo "[OK] Modal authentication complete."
fi

# --------------------------------------------------------------------------
# 3. Create Modal secrets for W&B and HuggingFace
# --------------------------------------------------------------------------
echo ""
echo "[STEP 2] Creating Modal secrets..."
echo "  You will need:"
echo "    - A Weights & Biases API key  (https://wandb.ai/authorize)"
echo "    - A HuggingFace access token  (https://huggingface.co/settings/tokens)"
echo ""

# --- W&B secret ---
read -rp "Enter your WANDB_API_KEY (paste and press Enter): " wandb_key
if [[ -z "$wandb_key" ]]; then
    echo "[WARN] Empty WANDB_API_KEY. Skipping wandb-secret creation."
    echo "       Training will fail if --report_to wandb is set in the shell script."
else
    modal secret create wandb-secret "WANDB_API_KEY=${wandb_key}" --force
    echo "[OK] Modal secret 'wandb-secret' created."
fi

echo ""

# --- HuggingFace secret ---
read -rp "Enter your HF_TOKEN (paste and press Enter): " hf_token
if [[ -z "$hf_token" ]]; then
    echo "[WARN] Empty HF_TOKEN. Skipping huggingface-secret creation."
    echo "       Model downloads for gated models (e.g., Gemma) will fail."
else
    modal secret create huggingface-secret "HF_TOKEN=${hf_token}" --force
    echo "[OK] Modal secret 'huggingface-secret' created."
fi

# --------------------------------------------------------------------------
# 4. Summary
# --------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Setup complete. You can now run:"
echo "  modal run modal_app.py::launch_train"
echo "  modal run modal_app.py::launch_eval"
echo "=========================================="
