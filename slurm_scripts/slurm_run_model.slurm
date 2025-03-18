#!/bin/bash
#SBATCH --job-name=ml_model_run
#SBATCH --output=../logs/slurm_%j.out
#SBATCH --error=../logs/slurm_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Activate your Python virtual environment (adjust the path if needed)
source ~/envs/ml/bin/activate

# Run the training script for a single model (e.g., 'my_method')
python train.py --model my_method --config experiments/config.yaml