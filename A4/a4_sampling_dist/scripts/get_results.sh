#!/usr/bin/env bash
# Script to download evaluation results from the Modal volume to the local machine.
# Usage: bash scripts/get_results.sh

set -euo pipefail

VOLUME_NAME="a4-sampling-results"
REMOTE_PATH="/sampling"
LOCAL_DIR="results"

# Ensure the parent results directory exists
mkdir -p "results"

echo "Syncing sampling results from Modal volume: $VOLUME_NAME at $REMOTE_PATH..."
# This will download the remote path into the local directory
modal volume get --force "$VOLUME_NAME" "$REMOTE_PATH" "$LOCAL_DIR"

echo "Results synced to $LOCAL_DIR/"
