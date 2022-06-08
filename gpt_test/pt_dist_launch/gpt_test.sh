#!/bin/bash

#SBATCH --wait-all-nodes=1
#SBATCH --job-name=pretraining
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexgre@ufl.edu
#SBATCH --nodes=69
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=1000gb
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --output=/red/gatortron-phi/gpt/gpt_test/log/gpt_n6_%j.out
#SBATCH --partition=gpu
#SBATCH --reservation=syngatortron
#SBATCH --exclusive
#SBATCH --exclude=c0906a-s29

ROOT=/red/gatortron-phi/gpt
VOCAB_FILE=${ROOT}/vocab/gpt2-vocab.json
MERGE_FILE=${ROOT}/vocab/gpt2-merges.txt
# DATA_PATH=/red/gatortron-phi/gpt/data/merged_uf_noftfy/uf_merged_noftfy #15604670
# DATA_PATH=/red/gatortron-phi/gpt/data/temp/gpt_training_uf_deid  #15604662
DATA_PATH=/red/gatortron-phi/gpt/data/uf_deid/uf_deid_gpt/note_txt_6_bin_text_document
CHECKPOINT_PATH=/red/gatortron-phi/gpt/gpt_test/model

# global-batch-size = micro-batch-size * GPU_num_per_node * node_num / tensor-model-parallel-size
MICRO_BATCH_SIZE=4
TENSOR_MODEL_PAR=2
GLOBAL_BATCH_SIZE=$((SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_TASK * MICRO_BATCH_SIZE / TENSOR_MODEL_PAR))
echo 'global batch size is set to '$GLOBAL_BATCH_SIZE

SAMPLE_PER_EPOCH=100886771   #262789535  # modify here
EPOCH_NUM=1 # modify here

L=24
H=4096
A=32

BERT_ARGS="--num-layers $L \
    --hidden-size $H \
    --num-attention-heads $A \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --train-samples $(($SAMPLE_PER_EPOCH * $EPOCH_NUM)) \
    --lr-decay-samples $(($SAMPLE_PER_EPOCH * $EPOCH_NUM * 867 / 1000)) \
    --lr-warmup-samples $(($SAMPLE_PER_EPOCH / 1000)) \
    --lr 1e-4 \
    --min-lr 1e-5 \
    --lr-decay-style linear \
    --seed 42 \
    --weight-decay 0.1 \
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
    --fp16"

OUTPUT_ARGS="--log-interval 100 \
    --save-interval 100000 \
    --eval-interval 500 \
    --eval-iters 100 \
    --checkpoint-activations"


# TRAINING_SCRIPT="/red/gatortron-phi/gpt/Megatron-LM-2022/pretrain_gpt.py"
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
PYTHON_PATH="singularity exec --nv /red/gatortron-phi/gpt/containers/pt2012.sif python"
# PYTHON_PATH="singularity exec --nv /red/gatortron-phi/gpt/containers/pt2112.sif python"

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

# system settings
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=1
#export NCCL_IB_TIMEOUT=22

PT_LAUNCH_SCRIPT=$(realpath "${PT_LAUNCH_UTILS_PATH}/run_pt_on_node.sh")
echo "Running \"$TRAINING_CMD\" on each node..."
srun "$PT_LAUNCH_SCRIPT" "$(realpath $PT_LAUNCH_UTILS_PATH)" \
    "$TRAINING_CMD" "$PYTHON_PATH"
