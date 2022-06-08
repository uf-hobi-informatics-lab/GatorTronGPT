#!/bin/bash


PT_LAUNCH_UTILS_PATH=$1
TRAINING_CMD=$2
PYTHON_PATH=$3

if [ -z "$PYTHON_PATH" ]
then
    PYTHON_PATH="python"
fi

# This should be the complete command to launch the per-node training run.
LAUNCH_CMD="$PYTHON_PATH \
        -m torch.distributed.launch \
              --nproc_per_node=$SLURM_GPUS_PER_TASK \
              --nnodes=$SLURM_JOB_NUM_NODES \
              --node_rank=$SLURM_NODEID \
              --master_addr=$PRIMARY \
              --master_port=$PRIMARY_PORT \
            $TRAINING_CMD"

source "${PT_LAUNCH_UTILS_PATH}/pt_multinode_helper_funcs.sh"
run_with_retry "$LAUNCH_CMD"


