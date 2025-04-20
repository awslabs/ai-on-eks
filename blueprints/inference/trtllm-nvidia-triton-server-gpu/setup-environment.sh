#!/bin/bash

# Create a directory to store model weights and download LLaMA model using Hugging Face CLI
mkdir llama

# Create directories to mount into the Docker container for engines and model files
mkdir triton_engines
mkdir triton_model_files

# change directory to /home/ubuntu
cd ~/

# Function to install Miniconda and create a new conda environment
install_miniconda() {
    CONDAENV=$1
    echo "Checking and installing Miniconda..."
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm ~/miniconda3/miniconda.sh
    source ~/miniconda3/bin/activate

    echo "Initializing conda and creating new environment..."
    conda create -n $CONDAENV python=3.12 -y
    eval "$(conda shell.bash hook)"
    conda activate $CONDAENV
}

# Function to install docker
install_docker() {
    echo "Checking and installing Docker..."
    if ! command -v docker &> /dev/null; then
        echo "Docker is not installed. Installing Docker..."
        sudo apt-get install docker -y
        sudo service docker start
        sudo usermod -aG docker $(whoami)
        # newgrp docker removed to prevent script interruption
    else
        echo "Docker is already installed."
    fi
}

# Function to install a package using pip
install_package() {
    PACKAGE=$1
    echo "Checking for $PACKAGE..."
    if ! python -c "import $PACKAGE" &> /dev/null; then
        echo "Installing $PACKAGE..."
        pip install $PACKAGE
    else
        echo "$PACKAGE is already installed."
    fi
}

# Function to install kubectl
install_kubectl() {
    echo "Installing kubectl..."
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl.sha256"
    echo "$(cat kubectl.sha256)  kubectl" | sha256sum --check
    sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
    kubectl version --client
}

# Function to install Terraform
install_terraform() {
    echo "Installing Terraform..."
    sudo apt-get update && sudo apt-get install -y gnupg software-properties-common
    wget -O- https://apt.releases.hashicorp.com/gpg | \
    gpg --dearmor | \
    sudo tee /usr/share/keyrings/hashicorp-archive-keyring.gpg > /dev/null
    gpg --no-default-keyring \
    --keyring /usr/share/keyrings/hashicorp-archive-keyring.gpg \
    --fingerprint
    
    echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] \
    https://apt.releases.hashicorp.com $(lsb_release -cs) main" | \
    sudo tee /etc/apt/sources.list.d/hashicorp.list
    
    sudo apt update
    sudo apt-get install terraform
    terraform -help
}

# Function to install AWS CLI v2
install_aws_cli() {
    echo "Installing AWS CLI v2..."
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install
    rm awscliv2.zip
    rm -rf aws
    echo "AWS CLI v2 installed successfully."
}

# Function to install tensorrt-backend
install_triton_tensorrtllm_backend() {
    echo "Installing Triton TensorRT-LLM backend..."
    git clone https://github.com/triton-inference-server/tensorrtllm_backend.git --branch v0.17.0
    cd tensorrtllm_backend

    echo "Installing Git LFS (used to fetch large files like model weights)..."
    sudo apt-get update
    sudo apt-get install git-lfs -y --no-install-recommends
    git lfs install

    echo "Downloading all submodules (required for the backend to function properly)..."
    git submodule update --init --recursive
}

install_miniconda ai-on-eks
install_docker
install_kubectl
install_terraform
install_aws_cli
echo "Installation of prerequisites complete."

sudo apt-get install jq
install_package boto3
install_package tritonclient[all]
install_package pandas
install_package numpy
install_package transformers
install_package huggingface_hub[cli]
echo "Installation of python dependencies complete."

install_triton_tensorrtllm_backend
echo "Environment setup complete."



