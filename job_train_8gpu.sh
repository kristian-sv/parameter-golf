#!/bin/bash
#SBATCH --account=nn11127k
#SBATCH --partition=accel
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --job-name=param-golf-8gpu
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

set -euo pipefail

# Load the NRIS PyTorch module for env vars (aliases don't expand in scripts)
module --quiet purge
module load NRIS/GPU PyTorch/2.8.0

PROJECT_DIR=/cluster/projects/nn11127k/ksv023
WORKDIR="$PROJECT_DIR/parameter-golf"
CONTAINER="$PYTORCH_CONTAINER_FILE"
OVERLAY="$PROJECT_DIR/.PyTorch/2.8.0/pytorch-overlay.img"

# Multi-node rendezvous via SLURM
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

# Helper to run commands inside the container
run_in_container() {
    srun apptainer exec --nv --fakeroot \
        --overlay "$OVERLAY":ro \
        --bind "$WORKDIR":"$WORKDIR" \
        --bind "$PROJECT_DIR/.cache":"$PROJECT_DIR/.cache" \
        --pwd "$WORKDIR" \
        --env DATA_PATH="$WORKDIR/data/datasets/fineweb10B_sp1024" \
        --env HF_HOME="$PROJECT_DIR/.cache/huggingface" \
        --env PYTHONNOUSERSITE=1 \
        --env TRITON_CACHE_DIR="$PROJECT_DIR/.cache/triton" \
        --env TORCHINDUCTOR_CACHE_DIR="$PROJECT_DIR/.cache/torchinductor" \
        --env TMPDIR="$PROJECT_DIR/.cache/tmp" \
        --env MASTER_ADDR="$MASTER_ADDR" \
        --env MASTER_PORT="$MASTER_PORT" \
        --env NCCL_DEBUG=INFO \
        "$CONTAINER" \
        "$@"
}

# Install extra deps into the persistent overlay (idempotent, single node only)
apptainer exec --nv --fakeroot \
    --overlay "$OVERLAY" \
    --bind "$WORKDIR":"$WORKDIR" \
    --bind "$PROJECT_DIR/.cache":"$PROJECT_DIR/.cache" \
    --pwd "$WORKDIR" \
    --env PYTHONNOUSERSITE=1 \
    "$CONTAINER" \
    pip install --no-cache-dir sentencepiece zstandard 2>/dev/null || true

# Run training across 2 nodes (4 GPUs each = 8 total)
run_in_container torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_id="$SLURM_JOB_ID" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    train_gpt.py
