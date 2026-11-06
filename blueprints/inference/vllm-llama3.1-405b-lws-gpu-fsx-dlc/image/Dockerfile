# Thin custom vLLM image = AWS Deep Learning Container (DLC) + Ray (+ Run:AI Model Streamer safety).
#
# Extends AWS's single-node vLLM DLC EKS example
# (https://aws.github.io/deep-learning-containers/vllm/deployment/eks/) to MULTI-NODE serving
# (TP x PP across nodes via LeaderWorkerSet + Ray).
#
# The current vLLM DLC (public.ecr.aws/deep-learning-containers/vllm:server-cuda) already bundles
# vLLM + EFA + aws-ofi-nccl (NCCL-over-EFA) AND the Run:AI Model Streamer — but NOT Ray. Multi-node
# vLLM (--distributed-executor-backend ray) REQUIRES Ray, so this layer adds Ray.
#
# Build + push to YOUR ECR (see README.md). DLC_BASE is the AWS DLC base — use the current tag from
# the gallery https://aws.github.io/deep-learning-containers/vllm/:
#   DLC_BASE=public.ecr.aws/deep-learning-containers/vllm:server-cuda   (public; no login)
#   private/canonical ECR URI: see the gallery "Image Access" page.
# Validated against the older 0.2x versioned DLC; re-validate against server-cuda.
ARG DLC_BASE
FROM ${DLC_BASE}

# Ray is required for multi-node (the DLC does not bundle it). The Run:AI streamer is already in the
# current DLC, so this --no-deps install is a harmless no-op there and a safety net for older/custom
# bases; --no-deps prevents it from changing the image's torch/vllm. Pin ray to a vLLM-compatible range.
RUN set -eux; \
    pip install --no-cache-dir --no-deps runai-model-streamer runai-model-streamer-s3; \
    pip install --no-cache-dir "ray[default]>=2.40,<2.50"; \
    python -c "import vllm, torch, ray; print('OK vllm', vllm.__version__, '| torch', torch.__version__, '| ray', ray.__version__)"
