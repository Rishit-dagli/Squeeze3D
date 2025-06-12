#!/bin/bash

set -e

echo "Checking system requirements..."

# Check CUDA version
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' || echo "")
if [[ -n "$CUDA_VERSION" ]] && [[ "${CUDA_VERSION%.*}" -ge 12 ]] && [[ "${CUDA_VERSION#*.}" -ge 1 || "${CUDA_VERSION%.*}" -gt 12 ]]; then
    echo "✓ CUDA $CUDA_VERSION"
else
    echo "⚠ WARNING: CUDA 12.1+ required (found: ${CUDA_VERSION:-not installed})"
fi

# Check Conda availability
if ! command -v conda &> /dev/null; then
    echo "⚠ WARNING: Conda is not installed. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create conda environment
conda create -n squeeze3d python=3.11 -y

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate squeeze3d

# Install pip and upgrade
pip install --upgrade pip

# Install PyTorch
conda install -y pytorch=2.4.1 torchvision=0.19.1 torchaudio=2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install packaging==24.1 wheel ninja ninja2 numpy==1.26.4 datasets==2.21.0
pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.1_cu121.html
pip install git+https://github.com/NVlabs/nvdiffrast git+https://github.com/KinglittleQ/torch-batch-svd
while IFS= read -r line || [ -n "$line" ]; do
    if [[ -z "$line" ]] || [[ "$line" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    echo "Installing: $line"
    pip install "$line" || echo "Failed to install: $line (continuing...)"
done < requirements-full.txt

conda uninstall pytorch torchvision torchaudio
conda install -y pytorch=2.4.1 torchvision=0.19.1 torchaudio=2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install numpy==1.26.4
pip install flash-attn --no-build-isolation

pip install -e .
