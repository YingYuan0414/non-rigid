#!/bin/bash

# This should take in 4 arguments:
# 1. the index of which GPU to use
# 2. model type
# 3. WANDB mode
# 4. dataset name
# 5. the rest of the arguments for the train.py script

# Example usage:
#./train_gmm.sh 0 rpdiff dataset.rpdiff_task_name=mug_rack_med_multi dataset.rpdiff_task_type=task_name_mug_on_rack_multi dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 resources.num_workers=8

# Resuming from a crashed run:
#./train_rigid.sh 0 ddrd_flow_separate rpdiff_fit online checkpoint.run_id=k8iy8vfo checkpoint.local_ckpt='/home/lyuxing/Desktop/tax3d_upgrade/scripts/logs/train_rpdiff_feature_df_cross/2025-03-02/17-41-55/checkpoints/last.ckpt'

# New:
#./train_gmm.sh 1 rpdiff model.point_encoder=pn2 dataset.rpdiff_task_name=mug_rack_med_multi dataset.rpdiff_task_type=task_name_mug_on_rack_multi dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 training.batch_size=32 training.val_batch_size=16 dataset.pcd_scale=8.0 model.pcd_scale=8.0 resources.num_workers=16 task_name=mug_multi_med_rack uniform_loss=0.1 regularize_residual=0.0
#./train_gmm.sh 0 rpdiff model.point_encoder=pn2 dataset.rpdiff_task_name=book_on_bookshelf_double_view_rnd_ori dataset.rpdiff_task_type=task_name_book_in_bookshelf dataset.pcd_scale_factor=20 dataset.sample_size_anchor=1024 dataset.pcd_scale=12.0 model.pcd_scale=12.0 resources.num_workers=16 task_name=book_bookshelf uniform_loss=0.1 regularize_residual=0.0

GPU_INDEX=$1
DATASET_NAME=$2
shift
shift
COMMAND=$@

WANDB_MODE=$WANDB_MODE python train_gmm.py \
  model.rel_pos=True \
  model.point_encoder=pn2 \
  dataset=$DATASET_NAME \
  resources.gpus=[${GPU_INDEX}] \
  $COMMAND
