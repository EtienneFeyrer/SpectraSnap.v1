#!/bin/bash
set -e

# Load conda shell functions
source ~/miniforge3/etc/profile.d/conda.sh

# Solve GPU stack and create environment
conda env create -f requirements/cuda.yaml

# Activate environment
conda activate jestr

# Install pip packages
pip install -r requirements/requirements.txt

# Install PyG CUDA wheels
pip install \
  torch-scatter==2.1.2+pt21cu121 \
  torch-sparse==0.6.18+pt21cu121 \
  torch-cluster==1.6.3+pt21cu121 \
  torch-geometric==2.5.1 \
  -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Install torchdata via conda-forge not possible since linked to open-ssl 3.x not compatible with Pytorch 2.2.0
pip install --no-deps torchdata==0.7.1


# Install DGL CUDA 12
pip install --no-deps dgl==2.1.0 -f https://data.dgl.ai/wheels/cu121/repo.html

# Clean up
conda clean -all
pip cache purge
