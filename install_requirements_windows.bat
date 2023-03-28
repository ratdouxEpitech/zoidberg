@echo off

REM Create and activate conda environment
conda create -n try1 python=3.8 -y
call activate try1

REM Install CUDA Toolkit and cuDNN
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y

REM Install TensorFlow (below 2.11 due to GPU support on Windows Native)
python -m pip install "tensorflow<2.11"

REM Install numpy and gpustat
conda install numpy -y
pip install gpustat

REM Verify installation
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
