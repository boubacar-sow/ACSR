#!/bin/bash
#SBATCH --job-name=cs_visual_cross_modal         # Job name
#SBATCH --partition=gpu               # Take a node from the 'gpu' partition
#SBATCH --export=ALL                  # Export your environment to the compute node
#SBATCH --cpus-per-task=2             # Number of CPU cores requested
#SBATCH --gres=gpu:A40:1              # Number and type of GPUs requested
#SBATCH --mem=12G                     # Memory request; MB assumed if not specified
#SBATCH --time=14:30:00               # Time limit hrs:min:sec
#SBATCH --output=%x-%j.log            # Standard output and error log

echo "Starting knowledge distillation script ..."
python /scratch2/bsow/Documents/ACSR/src/acsr/decoding.py

# Log job completion
echo "Job finished at: $(date)"
