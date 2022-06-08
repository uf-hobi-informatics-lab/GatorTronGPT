#!/bin/sh

##### resource allocation
#SBATCH --job-name=merge_json_files    # Job name
#SBATCH --mail-type=FAIL               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=alexgre@ufl.edu   # Where to send mail  
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=128             # Use 1 core
#SBATCH --mem=2000gb                     # Memory limit
#SBATCH --time=48:00:00               # Time limit hrs:min:sec
#SBATCH --output=/red/gatortron-phi/gpt/logs/prep/merge_json_files_%j.out   # Standard output and error log
#SBATCH --gpus-per-task=1
#SBATCH --partition=gpu
##SBATCH --reservation=gatortron

data_root=/data/ai/text/data/thepile
export SINGULARITY_BINDPATH=$data_root

TO_MERGE_DIR=$data_root
MERGED_FILE=/red/gatortron-phi/gpt/data/merged_thepile_jsonl/merged_thepile.jsonl

pwd; hostname; date
echo "merge json files in "${TO_MERGE_DIR}" as one file at "${MERGED_FILE}

find ${TO_MERGE_DIR} -name '*.jsonl' -exec cat {} + > ${MERGED_FILE}

# root=/red/gatortron-phi/gpt
# CONTAINER=${root}/containers/py2103.sif

# singularity exec --nv $CONTAINER python extract_text_from_json.py -i $MERGED_FILE -o ../data/raw_uf_notes.txt

date