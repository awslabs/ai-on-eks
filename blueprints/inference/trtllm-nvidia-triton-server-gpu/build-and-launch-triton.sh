#!/bin/bash

# -------------------------------- # 
# Step 1: Build the TensorRT model
# -------------------------------- # 
# Define paths and convert Hugging Face model to TensorRT-LLM checkpoint
HF_LLAMA_MODEL=/llama/Llama-3.2-1B-Instruct
TENSORRTLLM_CKPT_PATH=/tmp/tensorrt_checkpoint/llama32/1b/
ENGINE_DIR=/engines
CONVERT_CHKPT_SCRIPT=/tensorrtllm_backend/tensorrt_llm/examples/llama/convert_checkpoint.py

# Convert HF model to TRTLLM checkpoint format using a single GPU and FP16.
python ${CONVERT_CHKPT_SCRIPT} \
--model_dir ${HF_LLAMA_MODEL} \
--output_dir ${TENSORRTLLM_CKPT_PATH} \
--dtype float16 \
--tp_size 1

# Build TensorRT engine from the converted checkpoint
trtllm-build --checkpoint_dir ${TENSORRTLLM_CKPT_PATH} \
--output_dir ${ENGINE_DIR} \
--gemm_plugin auto \
--kv_cache_type paged \
--multiple_profiles enable \
--use_paged_context_fmha enable \
--max_batch_size 128 \
--max_num_tokens 65536


# -------------------------------- # 
# Step 2: Set values in model files
# -------------------------------- # 
# Copy the default inflight_batcher_llm model structure to Triton model directory
cp -r /tensorrtllm_backend/all_models/inflight_batcher_llm/* /triton_model_files/

# Define common environment variables to be passed to fill_template script
FILL_TEMPLATE_SCRIPT=/tensorrtllm_backend/tools/fill_template.py
MODEL_FOLDER=/triton_model_files
TOKENIZER_DIR=${HF_LLAMA_MODEL}
MAX_BATCH_SIZE=128
LOGITS_DT=TYPE_FP32
TOKENIZER_TYPE=auto
DECOUPLED_MODE=false
STREAMING_POSTPROCESS_TOKENS_TOGETHER=True
INSTANCE_COUNT=1
MAX_QUEUE_DELAY_MS=1000
TRITON_BACKEND=tensorrtllm
BATCHING_STRATEGY=inflight_fused_batching
KV_CACHE_REUSE=True
ENCODER_IPF_DT=TYPE_FP16
EXCLUDE_INPUT_IN_OUTPUT=TRUE
ENABLE_CHUNKED_CONTEXT=TRUE
ENABLE_CONTEXT_FMHA_FP32_ACCUMULATION=TRUE
SCHEDULER=max_utilization

# Fill configuration for each Triton model component using template script
python ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/ensemble/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},logits_datatype:${LOGITS_DT}
python ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/preprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},tokenizer_type:${TOKENIZER_TYPE},triton_max_batch_size:${MAX_BATCH_SIZE},preprocessing_instance_count:${INSTANCE_COUNT}
python ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/postprocessing/config.pbtxt tokenizer_dir:${TOKENIZER_DIR},tokenizer_type:${TOKENIZER_TYPE},triton_max_batch_size:${MAX_BATCH_SIZE},postprocessing_instance_count:${INSTANCE_COUNT}
python ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},bls_instance_count:${INSTANCE_COUNT},accumulate_tokens:${STREAMING_POSTPROCESS_TOKENS_TOGETHER},logits_datatype:${LOGITS_DT}
python ${FILL_TEMPLATE_SCRIPT} -i ${MODEL_FOLDER}/tensorrt_llm/config.pbtxt triton_backend:${TRITON_BACKEND},triton_max_batch_size:${MAX_BATCH_SIZE},decoupled_mode:${DECOUPLED_MODE},engine_dir:${ENGINE_DIR},max_queue_delay_microseconds:${MAX_QUEUE_DELAY_MS},batching_strategy:${BATCHING_STRATEGY},logits_datatype:${LOGITS_DT},enable_kv_cache_reuse:${KV_CACHE_REUSE},encoder_input_features_data_type:${ENCODER_IPF_DT},exclude_input_in_output:${EXCLUDE_INPUT_IN_OUTPUT},enable_chunked_context:${ENABLE_CHUNKED_CONTEXT},enable_context_fmha_fp32_acc:${ENABLE_CONTEXT_FMHA_FP32_ACCUMULATION},batch_scheduler_policy:${SCHEDULER}


# -------------------------------- # 
# Step 3: Start the server
# -------------------------------- # 
# Launch Triton inference server using the configured model repository
python /tensorrtllm_backend/scripts/launch_triton_server.py \
--world_size=1 \
--model_repo=${MODEL_FOLDER}
