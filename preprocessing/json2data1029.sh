#!/bin/sh

##### resource allocation
#SBATCH --job-name=json2data   # Job name
#SBATCH --mail-type=ALL               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=alexgre@ufl.edu  # Where to send mail  
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=128           # Use 1 core
#SBATCH --mem=2000gb  
#SBATCH --time=144:00:00          # Time limit hrs:min:sec
#SBATCH --output=/red/gatortron-phi/gpt/logs/prep/j2d_%A_%a.out    # Standard output and error log
#SBATCH --array=10-29
#SBATCH --partition=gpu
##SBATCH --reservation=gatortron

root=/red/gatortron-phi/gpt
# data_root=/data/ai/text/data/thepile
data_root=/red/gatortron-phi/gpt/data/ThePile_raw_json/the-eye.eu/public/AI/pile/train

vocab=${root}/vocab/gpt2-vocab.json
merge_file=${root}/vocab/gpt2-merges.txt

CONTAINER=${root}/containers/py2103.sif

# export SINGULARITY_BINDPATH=$data_root

i=$SLURM_ARRAY_TASK_ID

DATA=${data_root}/${i}.jsonl
PREFIX=${root}/data/new_preprocessed_thepile/thepile_${i}_bin

singularity exec --nv $CONTAINER python ${root}/Megatron-LM-2022/tools/preprocess_data.py \
        --input $DATA \
        --json-keys text \
        --tokenizer-type GPT2BPETokenizer \
        --vocab-file $vocab \
        --merge-file $merge_file \
        --output-prefix $PREFIX \
        --dataset-impl mmap \
        --workers 120 \
        --append-eod \
        --log-interval 10000