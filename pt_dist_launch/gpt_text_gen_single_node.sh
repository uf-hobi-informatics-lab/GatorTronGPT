#!/bin/bash

#SBATCH --wait-all-nodes=1
#SBATCH --job-name=gpt_text_gen
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexgre@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=2000gb
#SBATCH --time=5-00:00:00
#SBATCH --output=/red/gatortron-phi/gpt/logs/text_gen/gpt_5b_deid_%j.out
#SBATCH --partition=gpu
#SBATCH --reservation=syngatortron
#SBATCH --exclusive

export OMP_NUM_THREADS=1

ROOT=/red/gatortron-phi/gpt

VOCAB_FILE=${ROOT}/vocab/gpt2-vocab.json
MERGE_FILE=${ROOT}/vocab/gpt2-merges.txt

# #5b 
# L=24
# H=4096
# A=32
# TENSOR_MODEL_PAR=2
# CHECKPOINT_PATH=${ROOT}/models/gpt_uf_deid_thepile_mimic_5b_bs4

# #20b
CHECKPOINT_PATH=${ROOT}/models/gpt_uf_thepile_mimic_20b
# CHECKPOINT_PATH=${ROOT}/models/gpt_uf_thepile_mimic_20b_ep1_release
L=44
H=6144
A=48
TENSOR_MODEL_PAR=8

# fn=section_sample_manual
# fn=prompts_from_two_selected_samples
# fn=new_mimic_1000
fn=new_prompts_from_PH_RPDR
# fn=new_wiki_1000
SAMPLE_FILE=${ROOT}/gpt_syn/gpt_samples/${fn}.txt
# OUTPUT_FILE=${ROOT}/gpt_syn/gpt_results/gpt_5b_gen_${fn}.txt

TEMPERATURE=1.2
TOPP=0.9
KEPP_PROMPT=1
OUT_SEQ=512
MBS=32
# OUTPUT_FILE=${ROOT}/gpt_syn/gpt_results/gpt_5b_deid_gen_${fn}_temp_${TEMPERATURE}_${TOPP}_${OUT_SEQ}_${MBS}.txt
# OUTPUT_FILE=${ROOT}/gpt_syn/gpt_results/gpt_5b_gen_${fn}_temp_${TEMPERATURE}_${TOPP}_${OUT_SEQ}.txt
OUTPUT_FILE=${ROOT}/gpt_syn/gpt_results/gpt_20b_gen_${fn}_temp_${TEMPERATURE}_${TOPP}_${OUT_SEQ}_${MBS}.txt

#--seed 42
DISTRIBUTED_ARGS="--nproc_per_node $SLURM_GPUS_PER_TASK \
                  --nnodes 1 \
                  --node_rank 0"

GPT_ARGS="--num-layers $L \
    --hidden-size $H \
    --num-attention-heads $A \
    --seq-length 2048 \
    --micro-batch-size $MBS \
    --max-position-embeddings 2048 \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --DDP-impl torch \
    --tensor-model-parallel-size $TENSOR_MODEL_PAR \
    --pipeline-model-parallel-size 1 \
    --tokenizer-type GPT2BPETokenizer \
    --load $CHECKPOINT_PATH \
    --fp16"


GEN_ARGS="--sample-output-file $OUTPUT_FILE \
      --sample-input-file $SAMPLE_FILE \
      --out-seq-length $OUT_SEQ \
      --temperature $TEMPERATURE \
      --keep_prompt $KEPP_PROMPT \
      --top_k 0 \
      --top_p $TOPP"

# PYTHON_PATH="singularity exec --nv ${ROOT}/containers/pt2012.sif python"
# TRAINING_SCRIPT="${ROOT}/scripts/text_gen.py"

PYTHON_PATH="singularity exec --nv /red/gatortron-phi/gpt/containers/pt2112.sif python"
TRAINING_SCRIPT="${ROOT}/scripts/text_gen_api.py"

TRAINING_CMD="-m torch.distributed.launch $DISTRIBUTED_ARGS $TRAINING_SCRIPT $GPT_ARGS $GEN_ARGS"

$PYTHON_PATH $TRAINING_CMD