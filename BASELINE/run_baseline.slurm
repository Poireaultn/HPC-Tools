#!/bin/bash
#SBATCH --job-name=baseline_bert
#SBATCH --output=logs_baseline_%j.out
#SBATCH --error=logs_baseline_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1          # Request exactly 1 A100 GPU
#SBATCH --cpus-per-task=32         # Request 8 CPU cores for the DataLoader
#SBATCH --mem=64G                  # Request 64 Go of RAM

# Activate the course's virtual environment
module load cesga/2022
module load python/3.10.8
source $STORE/mypython/bin/activate

# Move to the directory where the script was submitted
cd $SLURM_SUBMIT_DIR

# Run the Python script
python train_bert.py