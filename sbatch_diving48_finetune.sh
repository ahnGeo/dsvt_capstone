#!/bin/bash
#SBATCH --job-name svt-ktod-ft-head-2*16*1e-3
#SBATCH -w aurora-g2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=15G
#SBATCH --time 2-0
#SBATCH --partition batch_sw_ugrad
#SBATCH -o slurm/logs/diving48/%A-%x.out
#SBATCH -e slurm/logs/diving48/%A-%x.err

PROJECT_PATH="/data/ahngeo11/svt"
EXP_NAME="diving48-head-1e-3-ori"
DATASET="diving48"
DATA_PATH="/data/ahngeo11/svt/datasets/annotations"
CHECKPOINT="/data/ahngeo11/svt/checkpoints/kinetics400_vitb_ssl.pth"


cd "$PROJECT_PATH" || exit

if [ ! -d "checkpoints/$EXP_NAME" ]; then
  mkdir "checkpoints/$EXP_NAME"
fi

export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --master_port="$RANDOM" \
  eval_linear.py \
  --n_last_blocks 1 \
  --arch "vit_base" \
  --pretrained_weights "$CHECKPOINT" \
  --epochs 15 \
  --lr 1e-3 \
  --batch_size_per_gpu 16 \
  --num_workers 8 \
  --num_labels 48 \
  --dataset "$DATASET" \
  --output_dir "checkpoints/$EXP_NAME" \
  --opts \
  DATA.PATH_TO_DATA_DIR "${DATA_PATH}" \
  DATA.PATH_PREFIX "/local_datasets/diving48/rgb" \
  DATA.USE_FLOW False

