#!/bin/bash
#SBATCH --job-name=cuedspeech                # Job name
#SBATCH --partition=cpu                     # Take a node from the 'cpu' partition
#SBATCH --export=ALL                        # Export current environement to compute node
#SBATCH --cpus-per-task=20                   # Ask for 10 CPU cores
#SBATCH --mem=10G                            # Memory request; MB assumed if not specified
#SBATCH --time=10:00:00                     # Time limit hrs:min:sec
#SBATCH --output=%x-%j.log                  # Standard output and error log

echo "Starting coordinates extraction  ..."
python /scratch2/bsow/Documents/ACSR/src/acsr/decoding.py

# Log job completion
echo "Job finished at: $(date)"
