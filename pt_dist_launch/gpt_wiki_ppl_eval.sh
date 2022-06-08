#!/bin/bash

#SBATCH --wait-all-nodes=1
#SBATCH --job-name=gpt_wiki_ppl
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexgre@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1000gb
#SBATCH --time=1-00:00:00
#SBATCH --output=/red/gatortron-phi/gpt/logs/wiki_ppl/gpt_20b_deid_%j.out
#SBATCH --partition=gpu
#SBATCH --reservation=syngatortron
#SBATCH --exclusive


TASK="WIKITEXT103"

export OMP_NUM_THREADS=1

ROOT=/red/gatortron-phi/gpt

VALID_DATA=${ROOT}/data/downstream/wikitext-103/wiki.test.tokens  # wiki.valid.tokens
VOCAB_FILE=${ROOT}/vocab/gpt2-vocab.json
MERGE_FILE=${ROOT}/vocab/gpt2-merges.txt

# 5b
# CHECKPOINT_PATH=${ROOT}/models/gpt_uf_thepile_mimic_5b 
# CHECKPOINT_PATH=${ROOT}/models/gpt_uf_deid_thepile_mimic_5b_bs4
# L=24
# H=4096
# A=32
# TENSOR_MODEL_PAR=2

#20b
# CHECKPOINT_PATH=${ROOT}/models/gpt_uf_thepile_mimic_20b
# # CHECKPOINT_PATH=${ROOT}/models/gpt_uf_thepile_mimic_20b_ep2
# CHECKPOINT_PATH=${ROOT}/models/gpt_uf_deid_thepile_mimic_20b_adjust/
CHECKPOINT_PATH=${ROOT}/models/gpt_uf_deid_thepile_mimic_20b
L=44
H=6144
A=48
TENSOR_MODEL_PAR=8

DISTRIBUTED_ARGS="--nproc_per_node $SLURM_GPUS_PER_TASK \
                  --nnodes 1 \
                  --node_rank 0"

COMMON_TASK_ARGS="--num-layers $L \
                  --hidden-size $H \
                  --num-attention-heads $A \
                  --seq-length 2048 \
                  --max-position-embeddings 2048 \
                  --fp16 \
                  --vocab-file $VOCAB_FILE"

# PYTHON_PATH="singularity exec --nv /red/gatortron-phi/gpt/containers/pt2012.sif python"
# PYTHON_PATH="singularity exec --nv /red/gatortron-phi/gpt/containers/pt2112.sif python"
PYTHON_PATH="singularity exec --nv /red/gatortron-phi/gpt/containers/pt2113.sif python"

#Megatron-LM-2022 Megatron-LM-latest
$PYTHON_PATH -m torch.distributed.launch $DISTRIBUTED_ARGS ${ROOT}/Megatron-LM-latest/tasks/main.py \
       --task $TASK \
       $COMMON_TASK_ARGS \
       --valid-data $VALID_DATA \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file $MERGE_FILE \
       --load $CHECKPOINT_PATH \
       --micro-batch-size 8 \
       --tensor-model-parallel-size $TENSOR_MODEL_PAR \
       --pipeline-model-parallel-size 1 \
       --activations-checkpoint-method uniform \
       --log-interval 100 \
       --no-load-optim \
       --no-load-rng