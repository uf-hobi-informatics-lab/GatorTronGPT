#!/bin/bash

#SBATCH --wait-all-nodes=1
#SBATCH --job-name=gpt_pretraining
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexgre@ufl.edu
#SBATCH --nodes=70
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=1000gb
#SBATCH --time=7-00:00:00
#SBATCH --output=/red/gatortron-phi/gpt/logs/gpt_training_uf_deid_thepile_mimic_20b_new/gpt_20b_%j.out
#SBATCH --partition=gpu
#SBATCH --reservation=syngatortron
##SBATCH --exclusive
#SBATCH --exclude=c1101a-s29,c0906a-s29  #c1010a-s23 c1109a-s23, ,c0800a-s11,c1004a-s23


export OMP_NUM_THREADS=1
export CUDA_DEVICE_WAITS_ON_EXCEPTION=1

ROOT=/red/gatortron-phi/gpt
VOCAB_FILE=${ROOT}/vocab/gpt2-vocab.json
MERGE_FILE=${ROOT}/vocab/gpt2-merges.txt
# DATA_PATH=/red/gatortron-phi/gpt/data/gpt_training_data/gpt_training_uf_deid_new_mimic
DATA_PATH=/red/gatortron-phi/gpt/data/gpt_training_data/gpt_training_uf_deid_thepile_mimic
CHECKPOINT_PATH=${ROOT}/models/gpt_uf_deid_thepile_mimic_20b_new

# we need to be agree on MBS and MP
MICRO_BATCH_SIZE=4
TENSOR_MODEL_PAR=8
GRADIENT_ACCUMULATE_STEP=1
GLOBAL_BATCH_SIZE=$((SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_TASK * MICRO_BATCH_SIZE / TENSOR_MODEL_PAR * GRADIENT_ACCUMULATE_STEP))
echo 'global batch size is set to '$GLOBAL_BATCH_SIZE
SAMPLE_PER_EPOCH=265715298   #262789535  # modify here
EPOCH_NUM=1 # modify here

# --lr-decay-iters 200000 \
# --train-iters 300000 \
# --lr-warmup-fraction 0.01 \
# --override-lr-scheduler \
# --checkpoint-activations
# --no-load-rng \
# --no-load-optim \

# echo train-samples $(($SAMPLE_PER_EPOCH * $EPOCH_NUM))  lr-decay-samples $(($SAMPLE_PER_EPOCH * $EPOCH_NUM / 2)) lr-warmup-samples $(($SAMPLE_PER_EPOCH / 100))

L=44
H=6144
A=48
BERT_ARGS="--num-layers $L \
    --hidden-size $H \
    --num-attention-heads $A \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --train-samples $(($SAMPLE_PER_EPOCH * $EPOCH_NUM)) \
    --lr-decay-samples $(($SAMPLE_PER_EPOCH * $EPOCH_NUM * 870 / 1000)) \
    --lr-warmup-samples $(($SAMPLE_PER_EPOCH / 1000)) \
    --lr 1e-4 \
    --min-lr 1e-5 \
    --override-lr-scheduler \
    --lr-decay-style linear \
    --seed 1234 \
    --weight-decay 0.01 \
    --clip-grad 1.0 \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 970,29,1 \
    --DDP-impl torch \
    --tensor-model-parallel-size $TENSOR_MODEL_PAR \
    --pipeline-model-parallel-size 1 \
    --tokenizer-type GPT2BPETokenizer \
    --tensorboard-dir /red/gatortron-phi/gpt/logs/gpt_training_uf_deid_thepile_mimic_20b_new \
    --fp16"

OUTPUT_ARGS="--log-interval 500 \
    --save-interval 5000 \
    --eval-interval 5000 \
    --eval-iters 100 \
    --checkpoint-activations"


TRAINING_SCRIPT="/red/gatortron-phi/gpt/Megatron-LM-latest/pretrain_gpt.py"

TRAINING_CMD="$TRAINING_SCRIPT \
    $BERT_ARGS \
    $OUTPUT_ARGS \
    --save $(realpath $CHECKPOINT_PATH) \
    --load $(realpath $CHECKPOINT_PATH) \
    --data-path $(realpath $DATA_PATH)"

#
# Python location (if not provided, system default will be used).
#

# PYTHON_PATH="singularity exec --nv /red/gatortron-phi/gpt/containers/pt2012.sif python"
PYTHON_PATH="singularity exec --nv /red/gatortron-phi/gpt/containers/pt2113.sif python"


#
# The location of the Pytorch multi-node launch utilities.
#
PT_LAUNCH_UTILS_PATH=pt_dist_launch
#
# The remainder of this script should not require modification.
#
source "${PT_LAUNCH_UTILS_PATH}/pt_multinode_helper_funcs.sh"
#export NCCL_DEBUG=INFO
init_node_info

pwd; hostname; date

echo "Primary node: $PRIMARY"
echo "Primary TCP port: $PRIMARY_PORT"
echo "Secondary nodes: $SECONDARIES"

PT_LAUNCH_SCRIPT=$(realpath "${PT_LAUNCH_UTILS_PATH}/run_pt_on_node.sh")
echo "Running \"$TRAINING_CMD\" on each node..."
srun "$PT_LAUNCH_SCRIPT" "$(realpath $PT_LAUNCH_UTILS_PATH)" \
    "$TRAINING_CMD" "$PYTHON_PATH"