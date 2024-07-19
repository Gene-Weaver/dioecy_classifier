#!/bin/bash

# Initialize and update submodules
git submodule update --init --recursive
git submodule foreach git pull origin main

# Install main project requirements
pip install -r requirements.txt

# Install submodule requirements
pip install -r LeafMachine2/requirements.txt
pip install git+https://github.com/waspinator/pycococreator.git@fba8f4098f3c7aaa05fe119dc93bbe4063afdab8#egg=pycococreatortools
pip install pywin32 pycocotools>=2.0.5 opencv-contrib-python>=4.7.0.68
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install vit-pytorch==0.37.1

python LeafMachine2/test.py