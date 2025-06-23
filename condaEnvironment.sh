#!/bin/bash

# Check if conda is already installed
if ! command -v conda &>/dev/null; then
    echo "Conda is not installed. Installing Anaconda..."
    wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh -O anaconda.sh
    chmod +x anaconda.sh
    # Run the installer with automated input
    ./anaconda.sh <<EOF

q
yes
yes
EOF
    source ~/.bashrc
else
    echo "Conda is already installed."
fi

# Check if the environment already exists
if ! conda info --envs | grep -q "^Pokemon"; then
    echo "Creating conda environment 'Pokemon'..."
    conda create --yes --name Pokemon
else
    echo "Conda environment 'Pokemon' already exists."
fi

# Initialize conda for shell usage
eval "$(conda shell.bash hook)"

conda activate Pokemon
conda install --yes pip
pip install -r requirements.txt
conda install --yes nodejs
