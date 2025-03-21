import json
import re
from typing import AsyncGenerator
from fastapi import BackgroundTasks, FastAPI
from prometheus_client import make_asgi_app
from starlette.requests import Request
from starlette.responses import StreamingResponse, Response, JSONResponse
from starlette.routing import Mount
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from ray import serve
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse
import os
import logging

from huggingface_hub import login
# Environment and configuration setup
logger = logging.getLogger("ray.serve")
app = FastAPI()
@serve.deployment(name="deployment",
                  autoscaling_config={"min_replicas": 1, "max_replicas": 2},
                  )
class VLLMDeployment:
    def __init__(self, **kwargs):
        route = Mount("/metrics", make_asgi_app())
        # Workaround for 307 Redirect for /metrics
        route.path_regex = re.compile('^/metrics(?P<path>.*)$')
        app.routes.append(route)
        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        logger.info(f"token: {hf_token=}")
        if not hf_token:
            raise ValueError("HUGGING_FACE_HUB_TOKEN environment variable is not set")
        login(token=hf_token)
        logger.info("Successfully logged in to Hugging Face Hub")

        args = AsyncEngineArgs(
            model=os.getenv("MODEL_ID", "NousResearch/Llama-3.2-1B"),
            # Model identifier from Hugging Face Hub or local path.
            dtype="auto",
            # Automatically determine the data type (e.g., float16 or float32) for model weights and computations.
            gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.8")),
            # Percentage of GPU memory to utilize, reserving some for overhead.
            max_model_len=int(os.getenv("MAX_MODEL_LEN", "8192")),
            # Maximum sequence length (in tokens) the model can handle, including both input and output tokens.
            max_num_seqs=int(os.getenv("MAX_NUM_SEQ", "4")),
            # Maximum number of sequences (requests) to process in parallel.
            max_num_batched_tokens=int(os.getenv("MAX_NUM_BATCHED_TOKENS", self.max_model_len * self.max_num_seqs)),
            # Maximum number of tokens processed in a single batch across all sequences (max_model_len * max_num_seqs).
            trust_remote_code=True,
            # Allow execution of untrusted code from the model repository (use with caution).
            enable_chunked_prefill=False,
            # Disable chunked prefill to avoid compatibility issues with prefix caching.
            tokenizer_pool_size=int(os.getenv("TOKENIZER_POOL_SIZE", "4")),  # Number of tokenizer instances to handle concurrent requests efficiently.
            tokenizer_pool_type="ray",  # Pool type for tokenizers; 'ray' uses Ray for distributed processing.
            max_parallel_loading_workers=int(os.getenv("MAX_PARALLEL_LOADING_WORKERS", "2")),  # Number of parallel workers to load the model concurrently.
            pipeline_parallel_size=int(os.getenv("PIPELINE_PARALLEL_SIZE", "1")),
            # Number of pipeline parallelism stages; typically set to 1 unless using model parallelism.
            tensor_parallel_size=int(os.getenv("TENSOR_PARALLEL_SIZE", "1")),
            # Number of tensor parallelism stages; typically set to 1 unless using model parallelism.
            enable_prefix_caching=bool(os.getenv("ENABLE_PREFIX_CACHING", "true"))  # Enable prefix caching to improve performance for similar prompt prefixes.
        )

        self.engine = AsyncLLMEngine.from_engine_args(args)
        self.max_model_len = args.max_model_len
        logger.info(f"VLLM Engine initialized with max_model_len: {self.max_model_len}")

    async def stream_results(self, results_generator) -> AsyncGenerator[bytes, None]:
        num_returned = 0
        async for request_output in results_generator:
            text_outputs = [output.text for output in request_output.outputs]
            assert len(text_outputs) == 1
            text_output = text_outputs[0][num_returned:]
            ret = {"text": text_output}
            yield (json.dumps(ret) + "\n").encode("utf-8")
            num_returned += len(text_output)

    async def may_abort_request(self, request_id) -> None:
        await self.engine.abort(request_id)

    async def __call__(self, request: Request) -> Response:
        try:
            request_dict = await request.json()
        except json.JSONDecodeError:
            return JSONResponse(status_code=400, content={"error": "Invalid JSON in request body"})

        context_length = request_dict.pop("context_length", 8192)  # Default to 8k

        # Ensure context length is either 8k or 32k or 128k
        if context_length not in [8192, 32768, 131072]:
            context_length = 8192  # Default to 8k if invalid
        prompt = request_dict.pop("prompt")
        stream = request_dict.pop("stream", False)

        # Get model config and tokenizer
        model_config = await self.engine.get_model_config()
        tokenizer = await self.engine.get_tokenizer()

        input_token_ids = tokenizer.encode(prompt)
        input_tokens = len(input_token_ids)
        max_possible_new_tokens = min(context_length, model_config.max_model_len) - input_tokens
        max_new_tokens = min(request_dict.get("max_tokens", 8192), max_possible_new_tokens)

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=request_dict.get("temperature", 0.7),
            top_p=request_dict.get("top_p", 0.9),
            top_k=request_dict.get("top_k", 50),
            stop=request_dict.get("stop", None),
        )

        request_id = random_uuid()
        logger.info(f"Processing request {request_id} with {input_tokens} input tokens")

        results_generator = self.engine.generate(prompt, sampling_params, request_id)

        if stream:
            background_tasks = BackgroundTasks()
            # Using background_tasks to abort the request
            # if the client disconnects.
            background_tasks.add_task(self.may_abort_request, request_id)
            return StreamingResponse(
                self.stream_results(results_generator), background=background_tasks
            )

        # Non-streaming case
        final_output = None
        async for request_output in results_generator:
            if await request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.engine.abort(request_id)
                logger.warning(f"Client disconnected for request {request_id}")
                return Response(status_code=499)
            final_output = request_output

        assert final_output is not None
        prompt = final_output.prompt
        text_outputs = [prompt + output.text for output in final_output.outputs]
        ret = {"text": text_outputs}
        logger.info(f"Completed request {request_id}")
        return Response(content=json.dumps(ret))

deployment = VLLMDeployment.bind()
