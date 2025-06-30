#!/bin/bash
#SBATCH --account=uw-soc
#SBATCH --time=00:10:00
#SBATCH --partition=beartooth-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --job-name=mpi_s2025_gpu_dp
#SBATCH --output=mpi_s2025_gpu_dp-%j.out
#SBATCH --error=mpi_s2025_gpu_dp-%j.err

# Conda env setup
chmod +x setup_conda_env.sh
./setup_env.sh

# Load conda module if available and/or source Conda initialization
module load conda || source "$HOME/miniconda3/etc/profile.d/conda.sh" || {
    echo "Conda not found. Please adjust the path or module name."
    exit 1
}


conda activate MPI_S2025
# Test using the Data Parallel GPU mode
python testing.py --train --gpu