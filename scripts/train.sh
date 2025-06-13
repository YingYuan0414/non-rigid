#!/bin/bash

# This should take in 4 arguments:
# 1. the index of which GPU to use
# 2. model type
# 3. WANDB mode
# 4. the rest of the arguments for the train.py script

########## RPDiff ##########
### Mug_Rack_Easy_Single ### *PCD_Scale=15*
## Anchor Centroid ##
# TAX3D Models #
# ./train.sh 0 cross_flow online rpdiff dataset.rpdiff_task_name=mug_rack_easy_single dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 training.batch_size=32 training.val_batch_size=16 resources.num_workers=32
# ./train.sh 0 cross_point online rpdiff dataset.rpdiff_task_name=mug_rack_easy_single dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 training.batch_size=32 training.val_batch_size=16 resources.num_workers=32

## Noisy Goal Centroid ##
# TAX3D Models #
# ./train.sh 0 cross_flow online rpdiff model.pred_frame=noisy_goal model.noisy_goal_scale=1.0 dataset.rpdiff_task_name=mug_rack_easy_single dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 training.batch_size=32 training.val_batch_size=16 resources.num_workers=32
# ./train.sh 0 cross_point online rpdiff model.pred_frame=noisy_goal model.noisy_goal_scale=1.0 dataset.rpdiff_task_name=mug_rack_easy_single dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 training.batch_size=32 training.val_batch_size=16 resources.num_workers=32
# TAX3Dv2 Models #
# ./train.sh 0 tax3dv2 online rpdiff model.frame_type=mu model.pred_frame=noisy_goal model.noisy_goal_scale=1.0 model.diff_rotation_noise_scale=45 dataset.rpdiff_task_name=mug_rack_easy_single dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 training.batch_size=32 training.val_batch_size=16 resources.num_workers=32
# ./train.sh 0 tax3dv2 online rpdiff model.frame_type=fixed model.pred_frame=noisy_goal model.noisy_goal_scale=1.0 model.diff_rotation_noise_scale=45 dataset.rpdiff_task_name=mug_rack_easy_single dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 training.batch_size=32 training.val_batch_size=16 resources.num_workers=32

# Ablade
# ./train.sh 0 tax3dv2 online rpdiff model.frame_type=fixed model.pred_frame=noisy_goal model.noisy_goal_scale=1.0 model.diff_rotation_noise_scale=45 model.joint_encode=True model.feature=True model.point_encoder=pn2 model.pcd_scale=3.0 dataset.pcd_scale=3.0 dataset.rpdiff_task_name=mug_rack_easy_single dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 training.batch_size=32 training.val_batch_size=16 resources.num_workers=32
# singularity exec --nv -B /home/lyuxingh/code/tax3d_upgrade:/opt/lyuxingh/code/tax3d_upgrade -B /scratch/lyuxingh/data:/opt/lyuxingh/data -B /scratch/lyuxingh/logs:/opt/lyuxingh/logs /scratch/lyuxingh/singularity/tax3d_lyuxing.sif bash -c "cd /opt/lyuxingh/code/tax3d_upgrade/scripts && CUDA_VISIBLE_DEVICES=0 ./train.sh 0 tax3dv2 online rpdiff model.frame_type=fixed model.pred_frame=noisy_goal model.noisy_goal_scale=1.0 model.diff_rotation_noise_scale=45 model.joint_encode=True model.feature=True model.point_encoder=pn2 model.pcd_scale=3.0 dataset.pcd_scale=3.0 dataset.rpdiff_task_name=mug_rack_easy_single dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 training.batch_size=32 training.val_batch_size=16 resources.num_workers=16 dataset.data_dir=/opt/lyuxingh/data/rpdiff/data/task_demos/ log_dir=/opt/lyuxingh/logs"

### Mug_Rack_Med_Single ### *PCD_Scale=15*
## Anchor Centroid ##
# TAX3D Models #
# ./train.sh 0 cross_flow online rpdiff dataset.rpdiff_task_name=mug_rack_med_single dataset.train_dataset_size=1600 dataset.val_dataset_size=200 dataset.test_dataset_size=200 training.batch_size=16 training.val_batch_size=16 resources.num_workers=32
# ./train.sh 0 cross_point online rpdiff dataset.rpdiff_task_name=mug_rack_med_single dataset.train_dataset_size=1600 dataset.val_dataset_size=200 dataset.test_dataset_size=200 training.batch_size=16 training.val_batch_size=16 resources.num_workers=32

## Noisy Goal Centroid ##
# TAX3Dv2 Models #
# ./train.sh 0 tax3dv2 online rpdiff model.frame_type=mu model.pred_frame=noisy_goal model.noisy_goal_scale=1.0 model.diff_rotation_noise_scale=45 dataset.rpdiff_task_name=mug_rack_med_single dataset.train_dataset_size=1600 dataset.val_dataset_size=200 dataset.test_dataset_size=200 training.batch_size=16 training.val_batch_size=16 resources.num_workers=32
# ./train.sh 0 tax3dv2 online rpdiff model.frame_type=fixed model.pred_frame=noisy_goal model.noisy_goal_scale=1.0 model.diff_rotation_noise_scale=45 dataset.rpdiff_task_name=mug_rack_med_single dataset.train_dataset_size=1600 dataset.val_dataset_size=200 dataset.test_dataset_size=200 training.batch_size=16 training.val_batch_size=16 resources.num_workers=32

### Mug_Rack_Med_Multi ### *PCD_Scale=15, Sample_Size_Anchor=1024*
## Anchor Centroid ##
# TAX3D Models #
# ./train.sh 0 cross_flow online rpdiff dataset.rpdiff_task_name=mug_rack_med_multi dataset.rpdiff_task_type=task_name_mug_on_rack_multi dataset.sample_size_anchor=1024 dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 training.batch_size=32 training.val_batch_size=16 resources.num_workers=32
# ./train.sh 0 cross_point online rpdiff dataset.rpdiff_task_name=mug_rack_med_multi dataset.rpdiff_task_type=task_name_mug_on_rack_multi dataset.sample_size_anchor=1024 dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 training.batch_size=32 training.val_batch_size=16 resources.num_workers=32

# dataset.rpdiff_task_name=mug_rack_med_multi dataset.rpdiff_task_type=task_name_mug_on_rack_multi dataset.sample_size_anchor=1024 dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 training.batch_size=32 training.val_batch_size=16

## Noisy Goal Centroid ##
# TAX3D Models #
# ./train.sh 0 cross_flow online rpdiff model.pred_frame=noisy_goal model.noisy_goal_scale=1.0 dataset.rpdiff_task_name=mug_rack_med_multi dataset.rpdiff_task_type=task_name_mug_on_rack_multi dataset.sample_size_anchor=1024 dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 training.batch_size=32 training.val_batch_size=16 resources.num_workers=32
# ./train.sh 0 cross_point online rpdiff model.pred_frame=noisy_goal model.noisy_goal_scale=1.0 dataset.rpdiff_task_name=mug_rack_med_multi dataset.rpdiff_task_type=task_name_mug_on_rack_multi dataset.sample_size_anchor=1024 dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 training.batch_size=32 training.val_batch_size=16 resources.num_workers=32
# TAX3Dv2 Models
# ./train.sh 0 tax3dv2 online rpdiff model.frame_type=mu model.pred_frame=noisy_goal model.noisy_goal_scale=1.0 model.diff_rotation_noise_scale=45 dataset.rpdiff_task_name=mug_rack_med_multi dataset.rpdiff_task_type=task_name_mug_on_rack_multi dataset.sample_size_anchor=1024 dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 training.batch_size=32 training.val_batch_size=16 resources.num_workers=32
# ./train.sh 0 tax3dv2 online rpdiff model.frame_type=fixed model.pred_frame=noisy_goal model.noisy_goal_scale=1.0 model.diff_rotation_noise_scale=45 dataset.rpdiff_task_name=mug_rack_med_multi dataset.rpdiff_task_type=task_name_mug_on_rack_multi dataset.sample_size_anchor=1024 dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 training.batch_size=32 training.val_batch_size=16 resources.num_workers=32

### Mug_Rack_Hard_Multi ### *PCD_Scale=15, Sample_Size_Anchor=1024*
## Anchor Centroid ##
# TAX3D Models #
# ./train.sh 0 cross_flow online rpdiff dataset.rpdiff_task_name=mug_on_rack_multi_large_proc_gen_demos dataset.rpdiff_task_type=task_name_mug_on_rack_multi dataset.sample_size_anchor=1024 dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 training.batch_size=32 training.val_batch_size=16 resources.num_workers=32
# ./train.sh 0 cross_point online rpdiff dataset.rpdiff_task_name=mug_on_rack_multi_large_proc_gen_demos dataset.rpdiff_task_type=task_name_mug_on_rack_multi dataset.sample_size_anchor=1024 dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 training.batch_size=32 training.val_batch_size=16 resources.num_workers=32

## Noisy Goal Centroid ##
# TAX3D Models #
# ./train.sh 0 cross_flow online rpdiff model.pred_frame=noisy_goal model.noisy_goal_scale=1.0 dataset.rpdiff_task_name=mug_on_rack_multi_large_proc_gen_demos dataset.rpdiff_task_type=task_name_mug_on_rack_multi dataset.sample_size_anchor=1024 dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 training.batch_size=32 training.val_batch_size=16 resources.num_workers=32
# ./train.sh 0 cross_point online rpdiff model.pred_frame=noisy_goal model.noisy_goal_scale=1.0 dataset.rpdiff_task_name=mug_on_rack_multi_large_proc_gen_demos dataset.rpdiff_task_type=task_name_mug_on_rack_multi dataset.sample_size_anchor=1024 dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 training.batch_size=32 training.val_batch_size=16 resources.num_workers=32
# TAX3Dv2 Models
# ./train.sh 0 tax3dv2 online rpdiff model.frame_type=mu model.pred_frame=noisy_goal model.noisy_goal_scale=1.0 model.diff_rotation_noise_scale=45 dataset.rpdiff_task_name=mug_on_rack_multi_large_proc_gen_demos dataset.rpdiff_task_type=task_name_mug_on_rack_multi dataset.sample_size_anchor=1024 dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 training.batch_size=32 training.val_batch_size=16 resources.num_workers=32
# ./train.sh 0 tax3dv2 online rpdiff model.frame_type=fixed model.pred_frame=noisy_goal model.noisy_goal_scale=1.0 model.diff_rotation_noise_scale=45 dataset.rpdiff_task_name=mug_on_rack_multi_large_proc_gen_demos dataset.rpdiff_task_type=task_name_mug_on_rack_multi dataset.sample_size_anchor=1024 dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 training.batch_size=32 training.val_batch_size=16 resources.num_workers=32

### Can_Cabnet ### *PCD_Scale=55, Sample_Size_Anchor=1024, Sample_Size_Acton=256*
## Anchor Centroid ##
# TAX3D Models #
# ./train.sh 0 cross_flow online rpdiff dataset.rpdiff_task_name=can_in_cabinet_stack dataset.rpdiff_task_type=task_name_stack_can_in_cabinet dataset.pcd_scale_factor=55 dataset.sample_size_anchor=1024 dataset.sample_size_action=256 dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 training.batch_size=32 training.val_batch_size=16 resources.num_workers=32

## Noisy Goal Centroid ##
# TAX3Dv2 Models
# ./train.sh 0 tax3dv2 online rpdiff model.frame_type=fixed model.pred_frame=noisy_goal model.noisy_goal_scale=1.0 model.diff_rotation_noise_scale=45 dataset.rpdiff_task_name=can_in_cabinet_stack dataset.rpdiff_task_type=task_name_stack_can_in_cabinet dataset.pcd_scale_factor=55 dataset.sample_size_anchor=1024 dataset.sample_size_action=256 dataset.train_dataset_size=3200 dataset.val_dataset_size=400 dataset.test_dataset_size=400 training.batch_size=32 training.val_batch_size=16 resources.num_workers=32


### Book_Bookshelf ### *PCD_Scale=20, Sample_Size_Anchor=1024*
## Anchor Centroid ##
# TAX3D Models #
# ./train.sh 0 cross_flow online rpdiff dataset.rpdiff_task_name=book_on_bookshelf_double_view_rnd_ori dataset.rpdiff_task_type=task_name_book_in_bookshelf dataset.pcd_scale_factor=20 dataset.sample_size_anchor=1024 dataset.train_dataset_size=1600 dataset.val_dataset_size=200 dataset.test_dataset_size=200 training.batch_size=16 training.val_batch_size=16 resources.num_workers=32

## Noisy Goal Centroid ##
# TAX3Dv2 Models
# ./train.sh 0 tax3dv2 online rpdiff model.frame_type=fixed model.pred_frame=noisy_goal model.noisy_goal_scale=1.0 model.diff_rotation_noise_scale=45 dataset.rpdiff_task_name=book_on_bookshelf_double_view_rnd_ori dataset.rpdiff_task_type=task_name_book_in_bookshelf dataset.pcd_scale_factor=20 dataset.sample_size_anchor=1024 dataset.train_dataset_size=1600 dataset.val_dataset_size=200 dataset.test_dataset_size=200 training.batch_size=16 training.val_batch_size=16 resources.num_workers=32

# Ablade
# singularity exec --nv -B /home/lyuxingh/code/tax3d_upgrade:/opt/lyuxingh/code/tax3d_upgrade -B /scratch/lyuxingh/data:/opt/lyuxingh/data -B /scratch/lyuxingh/logs:/opt/lyuxingh/logs /scratch/lyuxingh/singularity/tax3d_lyuxing.sif bash -c "cd /opt/lyuxingh/code/tax3d_upgrade/scripts && CUDA_VISIBLE_DEVICES=0 ./train.sh 0 tax3dv2 online rpdiff model.frame_type=fixed model.pred_frame=noisy_goal model.noisy_goal_scale=1.0 model.diff_rotation_noise_scale=45 model.joint_encode=True model.feature=True model.point_encoder=pn2 model.pcd_scale=40.0 dataset.pcd_scale=40.0 dataset.rpdiff_task_name=book_on_bookshelf_double_view_rnd_ori dataset.rpdiff_task_type=task_name_book_in_bookshelf dataset.pcd_scale_factor=20 dataset.sample_size_anchor=1024 dataset.train_dataset_size=1600 dataset.val_dataset_size=200 dataset.test_dataset_size=200 training.batch_size=16 training.val_batch_size=16 resources.num_workers=16 dataset.data_dir=/opt/lyuxingh/data/rpdiff/data/task_demos/ log_dir=/opt/lyuxingh/logs"

#------ Run On AutoBot with Singularity ------#
# singularity exec --nv -B /home/lyuxingh/code/tax3d_upgrade:/opt/lyuxingh/code/tax3d_upgrade -B /scratch/lyuxingh/data:/opt/lyuxingh/data -B /scratch/lyuxingh/logs:/opt/lyuxingh/logs /scratch/lyuxingh/singularity/tax3d_lyuxing.sif bash -c "cd /opt/lyuxingh/code/tax3d_upgrade/scripts && CUDA_VISIBLE_DEVICES=？ ./train.sh <YOUR TRAIN COMMAND HERE> dataset.data_dir=/opt/lyuxingh/data/rpdiff/data/task_demos/ log_dir=/opt/lyuxingh/logs"

#------ Resuming from a crashed run ------#
#./train.sh 0 cross_point_relative rpdiff False online checkpoint.run_id=k8iy8vfo checkpoint.local_ckpt='/home/lyuxing/Desktop/tax3d_upgrade/scripts/logs/train_rpdiff_feature_df_cross/2025-03-02/17-41-55/checkpoints/last.ckpt'

GPU_INDEX=$1
MODEL_TYPE=$2
WANDB_MODE=$3
DATASET_NAME=$4
shift
shift
shift
shift
COMMAND=$@
DATASET_PARAMS="dataset=$DATASET_NAME"

if [ $MODEL_TYPE == "cross_flow" ]; then
  echo "Training cross flow model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=df_cross model.type=flow"

elif [ $MODEL_TYPE == "cross_point" ]; then
  echo "Training cross point model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=df_cross model.type=point"

elif [ $MODEL_TYPE == "tax3dv2" ]; then
  echo "Training tax3dv2 model on dataset $DATASET_NAME with command: $COMMAND."

  MODEL_PARAMS="model=tax3dv2 model.type=point"

fi



WANDB_MODE=$WANDB_MODE python train.py \
  $MODEL_PARAMS \
  $DATASET_PARAMS \
  wandb.group=rigid \
  wandb.project=corl2025_tax3dv2 \
  resources.gpus=[${GPU_INDEX}] \
  $COMMAND


