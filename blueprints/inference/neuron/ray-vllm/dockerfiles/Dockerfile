# Use the base image
FROM rayproject/ray:2.47.0-py311

# Set environment variables to non-interactive (this prevents some prompts)
ENV DEBIAN_FRONTEND=non-interactive

# Switch to root to add Neuron repo and install necessary packages
USER root

# Set up the Neuron repository and install Neuron packages
RUN . /etc/os-release && \
    sudo echo "deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main" > /etc/apt/sources.list.d/neuron.list && \
    sudo wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | apt-key add - && \
    sudo apt-get update -y && \
    sudo apt-get install aws-neuronx-dkms aws-neuronx-collectives=2.* aws-neuronx-runtime-lib=2.* aws-neuronx-tools=2.* -y && \
    sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/*

# Switch back to a non-root user for the subsequent commands
USER $USER

# Set pip repository pointing to the Neuron repository and install required Python packages
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com && \
    python3 -m pip install --no-cache-dir wget awscli neuronx-cc==2.* torch-neuronx torchvision transformers-neuronx huggingface_hub 'cmake>=3.26' packaging 'setuptools-scm>=8' wheel jinja2

# Add Neuron path to PATH
ENV PATH=/opt/aws/neuron/bin:$PATH


# uninstall transformers-neuronx package explicitly to avoid version conflict
RUN python3 -m pip uninstall -y transformers-neuronx


RUN git clone -b v0.9.1 https://github.com/vllm-project/vllm.git && \
    cd vllm && \
    pip install -r requirements/neuron.txt && \
    VLLM_TARGET_DEVICE="neuron" pip install -e .

ENV VLLM_TARGET_DEVICE=neuron

# install transformers-neuronx package as an optional dependencies (for V0)
# FIXME: `--no-deps` argument is temporarily added to resolve transformers package version conflict
RUN python3 -m pip install transformers-neuronx --extra-index-url=https://pip.repos.neuron.amazonaws.com -U --no-deps

WORKDIR /home/ray
