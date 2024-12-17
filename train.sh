#!/bin/bash
#SBATCH --job-name=tomato_seg      # Job name
#SBATCH --output=job_%j.out        # Standard output and error log (%j expands to jobId)
#SBATCH --error=job_%j.err         # Error log
#SBATCH --time=00:02:00           # Time limit hrs:min:sec
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks-per-node=1       # Number of tasks per node
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --cpus-per-task=4         # CPU cores per task
#SBATCH --mem=16GB                # Memory limit
#SBATCH --partition=gpu           # GPU partition

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"

# Load necessary modules (modify according to your cluster's setup)
module purge
module load cuda/11.7
module load anaconda3/2022.05

# Activate your conda environment
#source activate py310

# Navigate to your project directory
#cd $SLURM_SUBMIT_DIR

# Run the training script
python train.py

# Print end time
echo "End time: $(date)"