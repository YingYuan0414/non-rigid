#!/bin/bash

# This should take in 3 arguments:
# 1. the index of which GPU to use
# 2. model checkpoint
# 3. the rest of the arguments for the eval script

# Example usage:
# ./tax3dv2_server.sh 0 kwcsrp5k gmm.gmm=1000 gmm.exp_name=book_bookshelf server.work_dir=/home/lyuxing/Desktop/tax3d_upgrade/scripts/tmp

# Get absolute path to the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_INDEX=$1
CHECKPOINT=$2
shift
shift
COMMAND=$@

echo "Evaluating model at checkpoint $CHECKPOINT with command: $COMMAND."
python "$SCRIPT_DIR/tax3dv2_server.py" \
    resources.gpus=[${GPU_INDEX}] \
    checkpoint.run_id=${CHECKPOINT} \
    wandb.group=rigid \
    wandb.project=corl2025_tax3dv2 \
    $COMMAND