#!/bin/bash

python3 /tensorrtllm_backend/scripts/launch_triton_server.py --world_size=1 --model_repo=/triton_model_files
