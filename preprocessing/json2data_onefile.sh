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
#SBATCH --output=/red/gatortron-phi/gpt/logs/prep/j2d_%j.out    # Standard output and error log
#SBATCH --partition=gpu
#SBATCH --reservation=gatortron

#srun --nodes=1 --partition=gpu --reservation=gatortron --gpus=1 --ntasks=1 --cpus-per-task=128 --mem 2000gb -t 3-00:00:00 --pty /bin/bash -i 


root=/red/gatortron-phi/gpt

vocab=${root}/vocab/gpt2-vocab.json
merge_file=${root}/vocab/gpt2-merges.txt

CONTAINER=${root}/containers/pt2012.sif

# data_root=/blue/data/ai/text/data/thepile
# ${root}/data/ThePile_raw_json/the-eye.eu/public/AI/pile/train/

# fid=Enron_Emails
# DATA=${root}/${fid}.jsonl
# PREFIX=${root}/${fid}_gpt_bin

# DATA=/blue/yonghui.wu/alexgre/data/m1.json
# PREFIX=${root}/m1_test

# cat m10lines to large one

DATA=/red/gatortron-phi/gpt/Enron_Emails.jsonl
PREFIX=${root}/Enron_Emails_2012
key=text
# key=NOTE_TEXT

export SINGULARITY_BINDPATH=$data_root
singularity exec --nv $CONTAINER python ${root}/Megatron-LM-2.5/tools/preprocess_data.py \
        --input $DATA \
        --json-keys $key \
        --tokenizer-type GPT2BPETokenizer \
        --vocab-file $vocab \
        --merge-file $merge_file \
        --output-prefix $PREFIX \
        --dataset-impl mmap \
        --workers 120 \
        --append-eod \
        --log-interval 10000