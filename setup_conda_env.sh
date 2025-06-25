#!/bin/bash

# Configuration
ENV_NAME="MPI_S2025"
PACKAGE_LIST="package-list.txt"
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

    # Initialize conda
    export PATH="$INSTALL_DIR/bin:$PATH"
    eval "$("$INSTALL_DIR/bin/conda" shell.bash hook)"
else
    echo "Conda found."
fi

# Create environment using package list
echo "Creating conda environment '$ENV_NAME'..."
conda create --name "$ENV_NAME" --file "$PACKAGE_LIST" -y

echo "Environment '$ENV_NAME' created successfully. Activate environment using 'conda activate MPI_S2025' "
