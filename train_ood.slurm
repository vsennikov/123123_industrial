#!/bin/bash
#SBATCH --account=EUHPC_D30_031
#SBATCH --job-name=123123
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc      # hackathon reservation (1 node per team)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4            # 1 GPU baseline; fair-share scales w/ this
#SBATCH --mem=480GB                  # 120GB * gpus-per-task
#SBATCH --cpus-per-task=32            # 8 * gpus-per-task
#SBATCH --time=02:00:00              # plenty for a ~7M model; raise for xl
#SBATCH --output=logs/%x_%j.out

# ---------------------------------------------------------------------------
# IMPORTANT (from the hackathon intro slides):
#  * Compute nodes have NO internet. Install everything on a LOGIN node first
#    (pixi / venv) and only RUN here. Do not pip-install inside this job.
#  * Use $SCRATCH for data + checkpoints. $WORK is off-limits for the hackathon.
#  * vocab.json must already exist (build it once on the login node).
# ---------------------------------------------------------------------------

set -euo pipefail
mkdir -p logs

PROJ=$SCRATCH/123123
cd "$PROJ"
export RUN="/leonardo/home/usertrain/a08trc10/.pixi/bin/pixi run"
RUN="${RUN:-}"

$RUN python3 train.py \
    --config baseline \
    --data data/mosfet.csv data/igbt.csv \
    --heldout data/heldout_ic.csv \
    --vocab vocab.json \
    --out "$PROJ/ckpt_ood_ic" \
    --epochs 30 \
    --batch_size 64