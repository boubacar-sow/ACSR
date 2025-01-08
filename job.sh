#!/bin/bash
#SBATCH --job-name=<name>             # Job name
#SBATCH --partition=gpu               # Take a node from the 'gpu' partition
#SBATCH --export=ALL                  # Export your environment to the compute node
#SBATCH --cpus-per-task=6             # Ask for 6 CPU cores
#SBATCH --gres=gpu:1                  # Ask for 1 GPU
#SBATCH --mem=100G                    # Memory request; MB assumed if unit not specified
#SBATCH --time=01:00:00               # Time limit hrs:min:sec
#SBATCH --output=%x-%j.log            # Standard output and error log

echo "Running job on $(hostname)"
echo "python: $(which python)"
echo "python-version $(python -V)"
echo "CUDA_DEVICE: $CUDA_VISIBLE_DEVICES"

python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'cuda device: {torch.cuda.current_device()}')"

echo "computation start $(date)"
# launch your computation
echo "computation end : $(date)"