#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Bash is working!"

# Initialize and update submodules
echo "Updating submodules..."
git submodule update --init --recursive
git submodule foreach git pull origin main

echo "Activating virtual environment..."
source .venv_dioecy/Scripts/activate

# Install main project requirements
# echo "Installing cmake, onnxruntime, onnxsim..."
# pip install cmake
# pip install setuptools cython onnx onnxruntime protobuf-compiler 
# pip.exe install grpcio-tools
# pip install onnxsim

echo "Installing main project requirements..."
pip install -r requirements.txt

# Install submodule requirements
echo "Installing LeafMachine2 requirements..."
pip install -r LeafMachine2/requirements.txt
pip install pywin32 pycocotools>=2.0.5 opencv-contrib-python>=4.7.0.68
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
echo "Setup complete!"