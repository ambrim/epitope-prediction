#!/bin/bash
#SBATCH --job-name=torchrun_job        # create a short name for your job
#SBATCH --nodes=1                      # node count
#SBATCH --ntasks=1                     # total number of tasks across all nodes
#SBATCH --cpus-per-task=4              # cpu-cores per task (1 since you have 1 process)
#SBATCH --gres=gpu:4                  # request 1 GPU
#SBATCH --mem-per-cpu=60G               # memory per CPU core (adjust as needed)
#SBATCH --time=20:00:00                # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin              # send email when job begins
#SBATCH --mail-type=end                # send email when job ends
#SBATCH --gpu-mps
#SBATCH --mail-user=ambri@princeton.edu
# Load any required modules or activate Conda environment
module purge
module load anaconda3/2023.9
conda activate epi
# export CUDA_LAUNCH_BLOCKING=1
# Change to the directory where your script is located
cd /scratch/gpfs/jiaweim/epitope
# Run your command with GPUs
python attempt7_150M_200k_default_accuracy_most.py