#!/bin/bash
#SBATCH --job-name=seq-to-seq      # Job name
#SBATCH --partition=gpu               # Take a node from the 'gpu' partition
#SBATCH --export=ALL                  # Export your environment to the compute node
#SBATCH --cpus-per-task=1             # Number of CPU cores requested
#SBATCH --gres=gpu:A40:1              # Number and type of GPUs requested
#SBATCH --mem=40G                     # Memory request; MB assumed if not specified
#SBATCH --time=2-00:00:00               # Time limit hrs:min:sec
#SBATCH --output=/scratch2/bsow/Documents/ACSR/src/acsr/logs/%x-%j.log            # Standard output and error log


python /scratch2/bsow/Documents/ACSR/src/acsr/seq_to_seq.py

# Log job completion
echo "Job finished at: $(date)"
