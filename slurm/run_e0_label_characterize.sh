#!/bin/bash
#SBATCH --job-name=smkvc_e0_label
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --signal=B:USR1@60
#SBATCH --output=/home/%u/workspace/self_managing_kvc/slurm_logs/e0_label_characterize_%j.out
#SBATCH --error=/home/%u/workspace/self_managing_kvc/slurm_logs/e0_label_characterize_%j.err
#SBATCH --gres=gpu:2
#SBATCH --constraint=A100_80GB|H100|H200|RTX_PRO_6000
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=40G
#SBATCH --time=4:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=skolawol@andrew.cmu.edu

# E0 label + characterize pipeline.
# Step 1: post-hoc eager attention pass → populates dead_end_flags per segment.
# Step 2: characterize.py → stats.json + figures/ for paper motivation section.
#
# Input:  data/math500_annotated.jsonl  (100 traces, already segmented)
# Output: data/math500_labeled.jsonl
#         data/e0_characterization/stats.json
#         data/e0_characterization/figures/

# ── Environment ────────────────────────────────────────────────────────────────
source $(conda info --base)/etc/profile.d/conda.sh
conda activate kvcache

export HF_HOME=/data/hf_cache/skolawol
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_CACHE=/data/hf_cache/skolawol/hub
export HF_DATASETS_CACHE=/data/hf_cache/skolawol/datasets
export TRANSFORMERS_CACHE=/data/hf_cache/skolawol

WORKDIR=/home/skolawol/workspace/self_managing_kvc
DATADIR=/data/user_data/skolawol/self_managing_kvc
cd "$WORKDIR"
mkdir -p slurm_logs "$DATADIR/data"

ANNOTATED=$DATADIR/data/math500_annotated.jsonl
LABELED=$DATADIR/data/math500_labeled.jsonl
OUTDIR=$DATADIR/data/e0_characterization

echo "=========================================="
echo "E0 — label_attention + characterize"
echo "Node: $(hostname)   Job: $SLURM_JOB_ID"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "Started: $(date)"
echo "Input:   $ANNOTATED"
echo "Output:  $LABELED  |  $OUTDIR/"
echo "=========================================="

# ── Step 1: Post-hoc attention labeling ───────────────────────────────────────
python scripts/label_attention.py \
    --input      "$ANNOTATED" \
    --output     "$LABELED" \
    --model      7b \
    --threshold  0.05 \
    --max_tokens 16384

EXIT=$?
echo "[Step 1] Done (exit $EXIT) at $(date)"
[ $EXIT -ne 0 ] && { echo "[ERROR] label_attention failed. Aborting."; exit $EXIT; }

# ── Step 2: Characterize ───────────────────────────────────────────────────────
echo ""
echo "[Step 2] Running characterize.py ..."
python scripts/characterize.py \
    --input      "$LABELED" \
    --output_dir "$OUTDIR"

EXIT2=$?
echo "[Step 2] Done (exit $EXIT2) at $(date)"
[ $EXIT2 -ne 0 ] && { echo "[ERROR] characterize failed."; exit $EXIT2; }

echo ""
echo "=========================================="
echo "Complete at $(date)"
echo "  Labeled:  $LABELED"
echo "  Stats:    $OUTDIR/stats.json"
echo "  Figures:  $OUTDIR/figures/"
echo "=========================================="
exit 0
