#!/bin/sh

##### resource allocation
#SBATCH --job-name=merge_data
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alexgre@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=1000gb
#SBATCH --gpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --output=/red/gatortron-phi/gpt/logs/prep/merge_%j.out
#SBATCH --partition=gpu 
##SBATCH --reservation=syngatortron

pwd; hostname; date

CONTAINER=/red/gatortron-phi/gpt/containers/py2103.sif

# IN=/red/gatortron-phi/gpt/data/uf_deid/uf_deid_gpt
# OUT=/red/gatortron-phi/gpt/data/uf_deid
# PREF=gpt_training_uf_deid

# IN=/red/gatortron-phi/gpt/data/new_preprocessed_thepile
# OUT=/red/gatortron-phi/gpt/data/merged_thepile_bin
# PREF=gpt_training_the_pile

IN=/red/gatortron-phi/gpt/data/temp
OUT=/red/gatortron-phi/gpt/data/gpt_training_data
PREF=gpt_training_uf_deid_new_mimic

VOB=/red/gatortron-phi/gpt/vocab/gpt2-vocab.json
MERGE=/red/gatortron-phi/gpt/vocab/gpt2-merges.txt

singularity exec $CONTAINER python /red/gatortron-phi/gpt/scripts/merge_mmap.py \
    --input $IN \
    --output $OUT \
    --output_prefix $PREF \
    --vocab_file $VOB \
    --merge_file $MERGE \
    --append_eod \
    --tokenizer_type GPT2BPETokenizer

date


