#!/bin/bash

# Configuration
ENV_NAME="MPI_S2025"
ENV_YML="environment.yml"  # Changed from package list txt to environment.yml
MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_URL="https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER"
INSTALL_DIR="$HOME/miniconda3"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."

    # Download the installer
    wget $MINICONDA_URL -O $MINICONDA_INSTALLER

    # Run the installer silently
    bash $MINICONDA_INSTALLER -b -p "$INSTALL_DIR"

    # Clean up
    rm $MINICONDA_INSTALLER

    # Initialize conda for the current shell session
    export PATH="$INSTALL_DIR/bin:$PATH"
    eval "$("$INSTALL_DIR/bin/conda" shell.bash hook)"
else
    echo "Conda found."
fi

# Add necessary channels (optional if already in environment.yml)
conda config --add channels conda-forge
conda config --add channels defaults

# Create environment from environment.yml
echo "Creating conda environment '$ENV_NAME' from $ENV_YML..."
conda env create -f "$ENV_YML" -n "$ENV_NAME"

echo "Environment '$ENV_NAME' created successfully."
echo "Activate environment using: conda activate $ENV_NAME"
