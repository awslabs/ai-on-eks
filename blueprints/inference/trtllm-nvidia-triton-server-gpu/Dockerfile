FROM nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3

COPY llama /llama
COPY triton_engines /engines
COPY triton_model_files /triton_model_files

RUN git clone https://github.com/triton-inference-server/tensorrtllm_backend.git /tensorrtllm_backend \
    && cd /tensorrtllm_backend \
    && apt-get update \
    && apt-get install git-lfs -y --no-install-recommends \
    && git submodule update --init --recursive \
    && git lfs install

COPY launch_triton_server.py /tensorrtllm_backend/scripts/launch_triton_server.py

COPY start.sh /start.sh
RUN chmod +x /start.sh

ENTRYPOINT ["/bin/bash", "/start.sh"]
