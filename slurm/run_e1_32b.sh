#!/bin/bash
#SBATCH --job-name=smkvc_e1_32b
#SBATCH --partition=general
#SBATCH --output=/home/%u/workspace/self_managing_kvc/slurm_logs/e1_32b_v%a_%j.out
#SBATCH --error=/home/%u/workspace/self_managing_kvc/slurm_logs/e1_32b_v%a_%j.err
#SBATCH --array=6,7,8,9
#SBATCH --gres=gpu:2
#SBATCH --constraint=A100_80GB|H100|H200|RTX_PRO_6000
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=40G
#SBATCH --time=16:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=skolawol@andrew.cmu.edu

# E1 zero-shot elicitation — 32B, all 5 prompt variants (array job).
# 2 GPUs: 32B model (~65GB) needs device_map="auto" across both.

source $(conda info --base)/etc/profile.d/conda.sh
conda activate kvcache

export HF_HOME=/data/hf_cache/skolawol
export HF_HUB_CACHE=/data/hf_cache/skolawol/hub
export HF_DATASETS_CACHE=/data/hf_cache/skolawol/datasets
export TRANSFORMERS_CACHE=/data/hf_cache/skolawol
export PYTORCH_ALLOC_CONF=expandable_segments:True

WORKDIR=/home/skolawol/workspace/self_managing_kvc
DATADIR=/data/user_data/skolawol/self_managing_kvc
cd "$WORKDIR"
mkdir -p slurm_logs "$DATADIR/data/e1"

V=$SLURM_ARRAY_TASK_ID
OUTPUT=$DATADIR/data/e1/v${V}_32b.jsonl

echo "=========================================="
echo "E1 — zero-shot elicitation | 32B | variant $V"
echo "Node: $(hostname)   Job: $SLURM_JOB_ID   Array: $SLURM_ARRAY_TASK_ID"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "Started: $(date)"
echo "Output:  $OUTPUT"
echo "=========================================="

python scripts/elicit_zero_shot.py \
    --model   32b \
    --variant $V \
    --problems data/math500_annotated.jsonl \
    --output  "$OUTPUT"

EXIT=$?
echo "Done (exit $EXIT) at $(date)"
exit $EXIT
