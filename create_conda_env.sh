#!/bin/sh


# Git lfs must be installed!!!!
conda create -y -n deep_knee python=3.6
conda install -y -n deep_knee numpy opencv scipy pyyaml cython
conda install -y -n deep_knee pytorch=0.3.1 -c soumith
conda install -y -n deep_knee git-lfs -c conda-forge

source activate deep_knee

pip install pip -U
pip install pydicom
pip install tqdm
pip install pillow
pip install torchvision
pip install termcolor