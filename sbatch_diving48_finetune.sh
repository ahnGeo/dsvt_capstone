#!/bin/bash
#SBATCH --job-name svt-ktod-ft-last-8*16*1e-4
#SBATCH -w aurora-g5
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=20G
#SBATCH --time 2-0
#SBATCH --partition batch_sw_ugrad
#SBATCH -o slurm/logs/diving48/%A-%x.out
#SBATCH -e slurm/logs/diving48/%A-%x.err

PROJECT_PATH="/data/ahngeo11/svt"
EXP_NAME="diving48-last-1e-4"
DATASET="diving48"
DATA_PATH="/data/ahngeo11/svt/datasets/annotations"
CHECKPOINT="/data/ahngeo11/svt/checkpoints/kinetics400_vitb_ssl.pth"


cd "$PROJECT_PATH" || exit

if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir "checkpoints/$EXP_NAME"
fi

MASTER_PORT=$((12000 + $RANDOM % 20000))

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.launch \
  --nproc_per_node=8 \       
  --master_port="$MASTER_PORT" \
  eval_linear.py \
  --n_last_blocks 1 \
  --arch "vit_base" \
  --pretrained_weights "$CHECKPOINT" \
  --epochs 50 \
  --lr 1e-4 \
  --batch_size_per_gpu 16 \
  --num_workers 8 \
  --num_labels 48 \
  --dataset "$DATASET" \
  --output_dir "checkpoints/$EXP_NAME" \
  --opts \
  DATA.PATH_TO_DATA_DIR "${DATA_PATH}" \
  DATA.PATH_PREFIX "/local_datasets/diving48_resized" \
  DATA.USE_FLOW False


