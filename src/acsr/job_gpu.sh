#!/bin/bash
#SBATCH --job-name=next-word     # Job name
#SBATCH --partition=gpu               # Take a node from the 'gpu' partition
#SBATCH --qos=gpu
#SBATCH --export=ALL                  # Export your environment to the compute node
#SBATCH --cpus-per-task=12             # Number of CPU cores requested
#SBATCH --gres=gpu:A100:1              # Number and type of GPUs requested
#SBATCH --mem=80G                     # Memory request; MB assumed if not specified
#SBATCH --output=/pasteur/appa/homes/bsow/ACSR/src/acsr/logs/%x-%j.log            # Standard output and error log


python3 /pasteur/appa/homes/bsow/ACSR/src/acsr/next_word_prediction.py

# Log job completion
echo "Job finished at: $(date)"


