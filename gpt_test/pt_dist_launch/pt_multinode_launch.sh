#!/bin/bash

#
# Script to launch a multi-node pytorch.distributed training run.
#
# (c) 2021, Brian J. Stucky
# UF Research Computing
#

#
# Resource allocation.
#
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=pretraining
#SBATCH --mail-type=NONE
#SBATCH --mail-user=alexgre@ufl.edu
#SBATCH --nodes=124
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=2000gb
#SBATCH --partition=hpg-ai
#SBATCH --reservation=HPGAI
# Enable the following to limit the allocation to a single SU.
## SBATCH --constraint=su7
#SBATCH --exclusive
#SBATCH --time=120:00:00
#SBATCH --output=new_log_uf_only/full_%j.out


#
# Training command specification.
#
#CHECKPOINT_PATH=/red/gatortron-phi/workspace/models/new_pretraining_checkpoint/4B_uf_full_uf30kcased
VOCAB_FILE=/red/gatortron-phi/workspace/data/vocabs/uf_text_97GB_cased_30k/uf_note_97GB_cased_30k_vocab.txt 
#DATA_PATH=/blue/gatortron/test-mn/data/uf1_TEXT_sentence
#DATA_PATH=/red/gatortron-phi/workspace/data/new_pretrain_data/uf_full_uf30kcased/uf_all_ufvocab_30k_cased_NOTE_TEXT_sentence
#DATA_PATH=/red/gatortron-phi/workspace/data/new_pretrain_data/uf_100g_uf30/uf_mimic_100GB_ufvocab_30k_cased_NOTE_TEXT_sentence
#DATA_PATH=/red/gatortron-phi/workspace/data/new_pretrain_data/uf_full_mimic_pubmed_wiki_uf30kcased/uf_full_wiki_pubmed_uf30kcased_TEXT

CHECKPOINT_PATH=/red/gatortron-phi/workspace/models/new_pretraining_checkpoint/4B_only_uf_uf30kcased
DATA_PATH=/red/gatortron-phi/workspace/data/new_pretrain_data/uf_full_uf30kcased/uf_all_ufvocab_30k_cased_NOTE_TEXT_sentence


#    --tokenizer-type BertWordPieceCase \
BERT_ARGS="--num-layers 48 \
    --hidden-size 2560 \
    --tokenizer-type BertWordPieceCase \
    --num-attention-heads 40 \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --lr 0.0001 \
    --lr-decay-iters 500000 \
    --seed 13 \
    --train-iters 1000000 \
    --min-lr 0.00001 \
    --lr-warmup-fraction 0.01 \
    --micro-batch-size 8 \
    --global-batch-size 3968 \
    --vocab-file $(realpath $VOCAB_FILE) \
    --split 949,50,1 \
    --DDP-impl torch \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 1 \
    --tensorboard-dir /red/gatortron-phi/workspace/new_log_uf_only \
    --fp16"

OUTPUT_ARGS="--log-interval 1000 \
    --save-interval 10000 \
    --eval-interval 5000 \
    --eval-iters 100 \
    --checkpoint-activations"

#TRAINING_SCRIPT="/blue/gatortron/test-mn/bjs_test/Megatron-LM/pretrain_bert.py"
TRAINING_SCRIPT="/red/gatortron-phi/workspace/scripts/Megatron-LM/pretrain_bert.py"
TRAINING_CMD="$TRAINING_SCRIPT \
    $BERT_ARGS \
    $OUTPUT_ARGS \
    --save $(realpath $CHECKPOINT_PATH) \
    --load $(realpath $CHECKPOINT_PATH) \
    --data-path $(realpath $DATA_PATH)"


#
# Python location (if not provided, system default will be used).
#
PYTHON_PATH="singularity exec --nv /red/gatortron-phi/workspace/containers/py2103.sif python"


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

