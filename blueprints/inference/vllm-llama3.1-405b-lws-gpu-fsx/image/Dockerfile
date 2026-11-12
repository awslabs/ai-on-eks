# Thin custom vLLM image = vllm/vllm-openai + Ray + Run:AI Model Streamer  (VLLM-OPENAI variant).
#
# Parallel to the DLC variant (which builds on the AWS DLC). The difference:
#   - DLC base      : ships vLLM + EFA drivers + aws-ofi-nccl (NCCL-over-EFA) but NO Ray.
#   - vllm-openai   : ships vLLM + Ray-compatible runtime but NO aws-ofi-nccl → inter-node NCCL = TCP.
# Both need the same thin layer (Ray + Run:AI streamer). FSx weight-LOADING rides EFA either way (it's a
# node/LNet property, not the container). Use this image when NCCL-over-EFA is NOT required.
#
# GOTCHA #4 (ai-on-eks R2 validation): the vllm/vllm-openai:v0.26.0 base ships `python3` (NO `python`
# symlink) and does NOT bundle Ray. So: verify with `python3`, and install Ray explicitly.
#
# Build + push to YOUR ECR — see README.md / build-kaniko.sh.
#   BASE_IMAGE default = vllm/vllm-openai:v0.26.0 (Docker Hub; validated pin — newer tags exist, override via --build-arg).
ARG BASE_IMAGE=vllm/vllm-openai:v0.26.0
FROM ${BASE_IMAGE}

RUN set -eux; \
    pip install --no-cache-dir --no-deps runai-model-streamer runai-model-streamer-s3; \
    pip install --no-cache-dir "ray[default]"; \
    python3 -c "import vllm, torch, ray; print('OK vllm', vllm.__version__, '| torch', torch.__version__, '| ray', ray.__version__)"
