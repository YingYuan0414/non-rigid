# Instructions for running this thing on Autobot.


0. Before you do anything, make sure you've built your docker image and pushed it to dockerhub!!!

1. ssh into autobot:

    ```
    ssh <SCS_username>@autobot.vision.cs.cmu.edu
    ```

    a. *YOU ONLY NEED TO DO THIS ONCE*: Add your wandb API key to your bashrc:

        ```bash
        echo 'export WANDB_API_KEY="your_api_key_here"' >> ~/.bashrc
        source ~/.bashrc
        ```

2. Find a node on http://autobot.vision.cs.cmu.edu/mtcmon/ which has open GPUs.

3. SSH into that node:

    ```
    ssh autobot-0-33
    ```

    a. *YOU ONLY NEED TO DO THIS ONCE*: Create some scratch directories for your data and logs.

        ```bash
        mkdir -p /scratch/$(whoami)/data
        mkdir -p /scratch/$(whoami)/logs
        ```
4. Run a training job like so. Don't worry about building or installing. You can modify the files here to map to whatever you want. In future iterations of this, we'll make this easier to do (aka by using a hydra singularity condfig file or something so you don't have to explictly map as arguments).

    You can also change which GPU you want access to using CUDA_VISIBLE_DEVICES below.

    ```bash
    SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 \
    SINGULARITYENV_WANDB_DOCKER_IMAGE=python-ml-project-template \
    singularity exec \
    --nv \
    --pwd /opt/$(whoami)/code \
    -B /scratch/$(whoami)/data:/opt/data \
    -B /scratch/$(whoami)/logs:/opt/logs \
    docker://beisner/python-ml-project-template \
    python scripts/train.py \
        dataset.data_dir=/opt/data \
        log_dir=/opt/logs
    ```



some ugly autobot commands for convenience until I've implemented some shell script functionality 
for autobot.

nstructions for running code on AUTOBOT:

Copying the data to the head node:
```
rsync -anv --exclude='*archive*' --exclude='*.gif'  proccloth/ eycai@autobot.vision.cs.cmu.edu:/project_data/held/eycai/data/proccloth
```

Copying data from head node to GPU node:
```
rsync -anv data/ autobot-0-25:/scratch/eycai/data
```

Copying the code to the head node:
```
rsync -anv --exclude='*scripts/logs/*' --exclude='.git/*' --exclude='*scripts/wandb*' --exclude='*.ckpt' --exclude='*notebooks/*' --exclude='third_party/*' non-rigid/ eycai@autobot.vision.cs.cmu.edu:code/non-rigid
```




TRAINING TAX3D FROM GPU NODE:
```
singularity exec --nv -B /home/eycai/code/non-rigid:/opt/eycai/code/non-rigid -B /scratch/eycai/data:/opt/eycai/data -B /scratch/eycai/logs:/opt/eycai/logs /scratch/eycai/singularity/tax3d.sif bash -c "cd /opt/eycai/code/non-rigid/scripts && CUDA_VISIBLE_DEVICES=0 ./train_deform.sh 0 cross_point online dedo dataset.data_dir=/opt/eycai/data/ dataset.train_size=400 model.joint_encode=True model.feature=True resources.num_workers=16"
```

TRAINING TAX3Dv2-fixed-frame FROM GPU NODE:
```
singularity exec --nv -B /home/eycai/code/non-rigid:/opt/eycai/code/non-rigid -B /scratch/eycai/data:/opt/eycai/data -B /scratch/eycai/logs:/opt/eycai/logs /scratch/eycai/singularity/tax3d.sif bash -c "cd /opt/eycai/code/non-rigid/scripts && CUDA_VISIBLE_DEVICES=0 ./train_deform.sh 0 tax3dv2_corl2025 online dedo dataset.data_dir=/opt/eycai/data/ dataset.hole=single dataset.train_size=400 resources.num_workers=16"
```

TRAINING TAX3Dv2-mu-frame FROM GPU NODE:
```
singularity exec --nv -B /home/eycai/code/non-rigid:/opt/eycai/code/non-rigid -B /scratch/eycai/data:/opt/eycai/data -B /scratch/eycai/logs:/opt/eycai/logs /scratch/eycai/singularity/tax3d.sif bash -c "cd /opt/eycai/code/non-rigid/scripts && CUDA_VISIBLE_DEVICES=0 ./train_deform.sh 0 tax3dv2 online dedo dataset.data_dir=/opt/eycai/data/ dataset.hole=single dataset.train_size=400 model.frame_type=mu model.joint_encode=True model.feature=True model.pred_frame=noisy_goal model.noisy_goal_scale=1.0 model.diff_rotation_noise_scale=45 resources.num_workers=16"
```

TRAINING DP3:
```
singularity exec --nv -B /home/eycai/code/non-rigid:/opt/eycai/code/non-rigid -B /scratch/eycai/data:/home/eycai/data -B /scratch/eycai/logs:/opt/eycai/logs /scratch/eycai/singularity/tax3d.sif bash -c "cd /opt/eycai/code/non-rigid/scripts && ./train_dedo_policy.sh dp3 dedo_proccloth none 1 0 easy"
```
