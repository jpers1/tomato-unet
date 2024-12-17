#!/bin/bash
#SBATCH --job-name=tomato_test     # Job name
#SBATCH --output=test_%j.out       # Standard output and error log
#SBATCH --error=test_%j.err        # Error log
#SBATCH --time=2:00:00            # Time limit hrs:min:sec (less than training)
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks-per-node=1       # Number of tasks per node
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --cpus-per-task=4         # CPU cores per task
#SBATCH --mem=64GB                # Memory limit
#SBATCH --partition=gpu           # GPU partition

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"

# Load necessary modules
module purge
module load CUDA/11.7.0
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
module load Python/3.10.4-GCCcore-11.3.0
module load matplotlib/3.5.2-foss-2022a
module load tqdm/4.64.1-GCCcore-12.2.0

# Install albumentations if not already installed
#pip install --user albumentations

# Print environment information
#echo "Python path: $(which python)"
#echo "Python version: $(python --version)"
#echo "CUDA version: $(nvcc --version)"

# Navigate to your project directory
#cd $SLURM_SUBMIT_DIR

# Run the test script
python test.py --model_path output_20241217_130332/checkpoints/best_model.pt

# Print end time
echo "End time: $(date)"
