#!/bin/bash
#SBATCH --job-name=array
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alexgre@ufl.edu    
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100gb  
#SBATCH --time=01:30:00
#SBATCH --output=array_%A-%a.log
#SBATCH --array=10-29
#SBATCH --partition=gpu
#SBATCH --reservation=gatortron

pwd; hostname; date
module load zstd
echo Task $SLURM_ARRAY_TASK_ID
FN=“/red/gatortron-phi/gpt/data/ThePile_raw_json/the-eye.eu/public/AI/pile/train/${SLURM_ARRAY_TASK_ID}.jsonl.zst”
zstd -d $FN
date