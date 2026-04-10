#!/bin/bash
#SBATCH --job-name=smkvc_e1_7b
#SBATCH --partition=general
#SBATCH --requeue
#SBATCH --signal=B:USR1@60
#SBATCH --output=/home/%u/workspace/self_managing_kvc/slurm_logs/e1_7b_v%a_%j.out
#SBATCH --error=/home/%u/workspace/self_managing_kvc/slurm_logs/e1_7b_v%a_%j.err
#SBATCH --array=1,6,7,8,9
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100_80GB|H100|H200|RTX_PRO_6000
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=24G
#SBATCH --time=8:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=skolawol@andrew.cmu.edu

# E1 zero-shot elicitation — 7B, all 5 prompt variants (array job).
# Each task runs one variant; all 5 run in parallel.
# SLURM_ARRAY_TASK_ID = variant number (1–5).

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
OUTPUT=$DATADIR/data/e1/v${V}_7b.jsonl

echo "=========================================="
echo "E1 — zero-shot elicitation | 7B | variant $V"
echo "Node: $(hostname)   Job: $SLURM_JOB_ID   Array: $SLURM_ARRAY_TASK_ID"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "Started: $(date)"
echo "Output:  $OUTPUT"
echo "=========================================="

python scripts/elicit_zero_shot.py \
    --model   7b \
    --variant $V \
    --problems data/math500_annotated.jsonl \
    --output  "$OUTPUT"

EXIT=$?
echo "Done (exit $EXIT) at $(date)"
exit $EXIT
