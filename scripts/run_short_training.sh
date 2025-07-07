#!/bin/bash

# Job name
#SBATCH --job-name=train
#SBATCH --partition=gpu

## GPU type constraint: A100-40 on xgph node or H100-96 on xgpi node
#SBATCH --constraint=xgpi # Use H100-96 GPU

## Request the appropriate GPU:
#SBATCH --gres=gpu:h100-47:1  # Use H100-96 GPU

## Set the runtime duration (adjust based on how long you expect the job to take)
#SBATCH --time=02:59:59  # HH:MM:SS (change as necessary)

# Resources: single task, single CPU core, 32 GB of memory
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G

## Log file names for output and error
#SBATCH --output=./logs/output_short_%j.slurmlog
#SBATCH --error=./logs/error_short_%j.slurmlog

# Display GPU status
nvidia-smi

# Activate virtual environment
source ".venv/bin/activate"

# Remove cache
rm -r __pycache__

if [ -z "$1" ]; then
    echo "Error: No Python file path provided."
    echo "Usage: $0 <path_to_python_file>"
    exit 1
fi

python "$1"
