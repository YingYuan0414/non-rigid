#!/bin/bash

# This should take in 4 arguments:
# 1. the index of which GPU to use
# 2. model type
# 3. WANDB mode
# 4. dataset name
# 5. the rest of the arguments for the train.py script

# Example usage:
# ./train_deform.sh 0 cross_flow offline dedo

# Resuming from a crashed run:
#./train_rigid.sh 0 ddrd_flow_separate rpdiff_fit online checkpoint.run_id=k8iy8vfo checkpoint.local_ckpt='/home/lyuxing/Desktop/tax3d_upgrade/scripts/logs/train_rpdiff_feature_df_cross/2025-03-02/17-41-55/checkpoints/last.ckpt'

GPU_INDEX=$1
CHECKPOINT=$2
shift
shift
COMMAND=$@

WANDB_MODE=$WANDB_MODE python eval_gmm.py \
  model.rel_pos=True \
  resources.gpus=[${GPU_INDEX}] \
  checkpoint.run_id=${CHECKPOINT} \
  $COMMAND
