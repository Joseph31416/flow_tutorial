#!/bin/bash
# Job name
#SBATCH --job-name=install_venv
#SBATCH --partition=gpu  

## GPU type constraint: A100-40 on xgph node or H100-96 on xgpi node
## #SBATCH --constraint=xgph # Use A100-40 GPU
#SBATCH --constraint=xgpi # Use H100-96 GPU

## Request the appropriate GPU:
## #SBATCH --gres=gpu:a100-40:1  # Use A100-40 GPU
#SBATCH --gres=gpu:h100-47:1  # Use H100-47 GPU


## Set the runtime duration (adjust based on how long you expect the job to take)
#SBATCH --time=00:14:59  # HH:MM:SS (change as necessary)

# Resources: single task, single CPU core, 20 GB of memory
#SBATCH --ntasks=1                          # Number of tasks (1 task)
#SBATCH --cpus-per-task=1                   # Number of CPU cores per task
#SBATCH --mem=32G                           # 20GB of memory per task

## Log file names for output and error
#SBATCH --output=./logs/install_venv.slurmlog
#SBATCH --error=./logs/error_install_venv.slurmlog

nvidia-smi

source ".venv/bin/activate"

rm -r __pycache__
pip install -r requirements.txt
# pip install -e .
