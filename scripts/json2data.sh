#!/bin/sh

##### resource allocation
#SBATCH --job-name=json2data   # Job name
#SBATCH --mail-type=ALL               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=alexgre@ufl.edu  # Where to send mail  
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=40           # Use 1 core
#SBATCH --mem=1000gb  
#SBATCH --time=144:00:00          # Time limit hrs:min:sec
#SBATCH --output=/red/gatortron-phi/gpt/logs/prep/j2d_%A_%a.out    # Standard output and error log
#SBATCH --array=0-19
#SBATCH --partition=gpu
#SBATCH --reservation=gatortron

root=/red/gatortron-phi/gpt

vocab=${root}/vocab/gpt2-vocab.json
merge_file=${root}/vocab/gpt2-merges.txt


CONTAINER=${root}/containers/py2103.sif


data_root=/data/ai/text/data/thepile


file_names=(ArXiv_sample.jsonl \
BookCorpus2_sample.jsonl \
Books3_sample.jsonl \
DM_Mathematics_sample.jsonl \
Enron_Emails.jsonl \
EuroParl_sample.jsonl \
FreeLaw_sample.jsonl \
Github_sample.jsonl \
Gutenberg_PG19_sample.jsonl \
HackerNews_sample.jsonl \
NIH_Exporter_sample.jsonl \
OpenSubtitles_sample.jsonl \
OpenWebText2_sample.jsonl \
Pile-CC_sample.jsonl \
PubMedAbstracts.jsonl \
PubMedCentral.jsonl \
StackExchange_sample.jsonl \
UbuntuIRC_sample.jsonl \
USPTO_Backgrounds_sample.jsonl \
YoutubeSubtitles_sample.jsonl)

output_prefix=(ArXiv_sample_gpt_bin \
BookCorpus2_sample_gpt_bin \
Books3_sample_gpt_bin \
DM_Mathematics_sample_gpt_bin \
Enron_Emails_gpt_bin \
EuroParl_sample_gpt_bin \
FreeLaw_sample_gpt_bin \
Github_sample_gpt_bin \
Gutenberg_PG19_sample_gpt_bin \
HackerNews_sample_gpt_bin \
NIH_Exporter_sample_gpt_bin \
OpenSubtitles_sample_gpt_bin \
OpenWebText2_sample_gpt_bin \
Pile-CC_sample_gpt_bin \
PubMedAbstracts_gpt_bin \
PubMedCentral_gpt_bin \
StackExchange_sample_gpt_bin \
UbuntuIRC_sample_gpt_bin \
USPTO_Backgrounds_sample_gpt_bin \
YoutubeSubtitles_sample_gpt_bin)


# for i in $(seq 0 19)
# do
#     echo $i
#     echo ${data_root}/${file_names[$i]}
#     echo ${root}/${output_prefix[$i]}
# done
export SINGULARITY_BINDPATH=$data_root

i=$SLURM_ARRAY_TASK_ID
DATA=${data_root}/${file_names[$i]}
PREFIX=${root}/data/preprocessed_thepile/${output_prefix[$i]}


singularity exec --nv $CONTAINER python ${root}/Megatron-LM-3860e99/tools/preprocess_data.py \
        --input $DATA \
        --json-keys text \
        --tokenizer-type GPT2BPETokenizer \
        --vocab-file $vocab \
        --merge-file $merge_file \
        --output-prefix $PREFIX \
        --dataset-impl mmap \
        --workers 40 \
        --append-eod \
        --log-interval 10000