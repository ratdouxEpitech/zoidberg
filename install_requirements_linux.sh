#!/bin/bash

# Create and activate conda environment
conda create -n try1 python=3.8 -y
source activate try1

# Install CUDA Toolkit
conda install -c conda-forge cudatoolkit=11.8.0 -y

# Install cuDNN and TensorFlow
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 "tensorflow==2.12.*"

# Set environment variables
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib

# Install numpy and gpustat
conda install numpy -y
pip install gpustat

# Verify installation
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
