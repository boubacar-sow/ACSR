#!/bin/bash
#SBATCH --job-name=no-lips
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --export=ALL
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:A100:1          # Request 2 GPUs of any type
#SBATCH --mem=3G
#SBATCH --output=/pasteur/appa/homes/bsow/ACSR/src/acsr/logs/grid_search/grid_search_no_lips_array_%A_%a.log
#SBATCH --array=1-6      # Total number of parameter combinations
#SBATCH --time=24:00:00                     # Time limit hrs:min:sec

# Define parameter ranges
alphas=(0.2 0.5 1.0)
hidden_dims=(64 128) 
n_layers=(2)
optimizers=("adam")
learning_rates=(1e-3)

# Calculate total combinations
total_combinations=$((${#alphas[@]} * ${#hidden_dims[@]} * ${#n_layers[@]} * ${#optimizers[@]} * ${#learning_rates[@]}))

# Map task ID to parameter combination
index=$(($SLURM_ARRAY_TASK_ID - 1))
alpha=${alphas[$((index % ${#alphas[@]}))]}
index=$((index / ${#alphas[@]}))
hidden_dim=${hidden_dims[$((index % ${#hidden_dims[@]}))]}
index=$((index / ${#hidden_dims[@]}))
layers=${n_layers[$((index % ${#n_layers[@]}))]}
index=$((index / ${#n_layers[@]}))
optimizer=${optimizers[$((index % ${#optimizers[@]}))]}
index=$((index / ${#optimizers[@]}))
lr=${learning_rates[$((index % ${#learning_rates[@]}))]}

# Log the allocated GPU type
echo "Allocated GPUs: $CUDA_VISIBLE_DEVICES"

# Run the Python script with the current parameters
python3 /pasteur/appa/homes/bsow/ACSR/src/acsr/decoding_grid_search_only_lips.py \
    --alpha $alpha \
    --encoder_hidden_dim $hidden_dim \
    --n_layers $layers \
    --optimizer $optimizer \
    --learning_rate $lr

echo "Job finished at: $(date)"