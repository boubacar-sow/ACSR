#!/bin/bash
#SBATCH --job-name=cuedspeech         # Job name
#SBATCH --partition=cpu                     # Take a node from the 'cpu' partition
#SBATCH --export=ALL                        # Export current environement to compute node
#SBATCH --cpus-per-task=1                   # Ask for 10 CPU cores
#SBATCH --mem=1G                            # Memory request; MB assumed if not specified
#SBATCH --time=10:00:00                     # Time limit hrs:min:sec
#SBATCH --output=jupyter-notebook-%J.out           # Standard output and error log

echo "Running via sbatch on $(hostname) on $(date)"

port=8003
node=$(hostname -s)
user=$(whoami)

# run jupyter notebook
jupyter notebook --ContentsManager.allow_hidden=True --no-browser --port=${port} --ip=${node}