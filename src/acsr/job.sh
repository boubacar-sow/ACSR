#!/bin/bash
#SBATCH --job-name=ipa_generation      # Job name
#SBATCH --partition=cpu                # Use the CPU partition
#SBATCH --export=ALL                   # Export current environment to the compute node
#SBATCH --cpus-per-task=4            # Request 10 CPU cores
#SBATCH --mem=2G                      # Request 1 GB of memory
#SBATCH --time=04:00:00                # Time limit: 4 hours
#SBATCH --output=%x-%j.log             # Standard output and error log

# Activate the Conda environment where MFA and Whisper are installed
# Run the Python script with multiprocessing
echo "Starting IPA generation..."
python /scratch2/bsow/Documents/ACSR/src/acsr/build_ipa_sentence_with_whisper_and_mfa.py

# Log job completion
echo "IPA generation completed."
echo "Job finished at: $(date)"