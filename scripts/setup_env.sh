#!/bin/bash

# Setup conda environment and install dependencies
ENV_NAME="trace"

echo "Setting up TRACE-CT conda environment..."

# Create conda environment with python 3.11 and pip
source $(conda info --base)/etc/profile.d/conda.sh
conda create -y -n $ENV_NAME python=3.11 pip
conda activate $ENV_NAME

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing main dependencies..."
pip install -r requirements.txt

echo "Installing dev dependencies..."
pip install -r requirements-dev.txt

# Install the package itself in editable mode
pip install -e .

echo "Setup complete. To activate the environment, run:"
echo "conda activate $ENV_NAME"
