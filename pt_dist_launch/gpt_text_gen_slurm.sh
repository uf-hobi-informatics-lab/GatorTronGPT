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
#SBATCH --partition=gpu
#SBATCH --reservation=syngatortron
#SBATCH --exclusive
#SBATCH --output=/red/gatortron-phi/gpt/logs/text_gen/array/large_scale_mimic_text_gen_%A_%a.out
#SBATCH --array=1-10

task_id=$SLURM_ARRAY_TASK_ID

export OMP_NUM_THREADS=1

ROOT=/red/gatortron-phi/gpt

VOCAB_FILE=${ROOT}/vocab/gpt2-vocab.json
MERGE_FILE=${ROOT}/vocab/gpt2-merges.txt

# #5b 
# L=24
# H=4096
# A=32
# TENSOR_MODEL_PAR=2
# CHECKPOINT_PATH=${ROOT}/models/gpt_uf_thepile_mimic_5b

# #20b
CHECKPOINT_PATH=${ROOT}/models/gpt_uf_thepile_mimic_20b
L=44
H=6144
A=48
TENSOR_MODEL_PAR=8

# fn=new_mimic_100
# fn=new_wiki_100
# fn=mimic_template_10
# fn=section_sample_manual
fn=mimic_hearder_prompts_100

SAMPLE_FILE=${ROOT}/gpt_syn/gpt_samples/${fn}.txt
TEMPERATURE=1.2
TOPP=0.9
OLEN=512
KEPP_PROMPT=0

# output_dir=${ROOT}/gpt_syn/gpt_results/gpt_5b_gen_${fn}_${TEMPERATURE}_${TOPP}
output_dir=${ROOT}/gpt_syn/gpt_results/gpt_20b_gen_${fn}_${TEMPERATURE}_${TOPP}_${OLEN}
mkdir -p $output_dir
OUTPUT_FILE=${output_dir}/gen_text_${task_id}.txt

random_seed=$(($task_id * 77)) # gen random seeds from array id

#--seed 42
DISTRIBUTED_ARGS="--nproc_per_node $SLURM_GPUS_PER_TASK \
                  --nnodes 1 \
                  --node_rank 0"

GPT_ARGS="--num-layers $L \
    --hidden-size $H \
    --num-attention-heads $A \
    --seq-length 2048 \
    --micro-batch-size 64 \
    --max-position-embeddings 2048 \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --DDP-impl torch \
    --tensor-model-parallel-size $TENSOR_MODEL_PAR \
    --pipeline-model-parallel-size 1 \
    --tokenizer-type GPT2BPETokenizer \
    --load $CHECKPOINT_PATH \
    --fp16\
    --seed $random_seed"

GEN_ARGS="--sample-output-file $OUTPUT_FILE \
      --sample-input-file $SAMPLE_FILE \
      --out-seq-length $OLEN \
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