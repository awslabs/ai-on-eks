import argparse
import asyncio
import json
import sys
import time  # Import the time module
from os import system
import numpy as np

import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import *
# from tritonclient.grpc import InferInput, InferRequestedOutput


def count_tokens(text):
    return len(text.split())


def create_request(
    prompt,
    stream,
    request_id,
    model_name,
    sampling_parameters,
    max_tokens,
):
    inputs = []

    # Required input
    prompt_data = np.array([[prompt.encode("utf-8")]], dtype=np.object_)
    inputs.append(grpcclient.InferInput("text_input", prompt_data.shape, "BYTES"))
    inputs[-1].set_data_from_numpy(prompt_data)

    # Required max_tokens
    max_tokens_data = np.array([[max_tokens]], dtype=np.int32)
    inputs.append(grpcclient.InferInput("max_tokens", max_tokens_data.shape, "INT32"))
    inputs[-1].set_data_from_numpy(max_tokens_data)

    # Optional stream
    stream_data = np.array([[stream]], dtype=bool)
    inputs.append(grpcclient.InferInput("stream", stream_data.shape, "BOOL"))
    inputs[-1].set_data_from_numpy(stream_data)

    # Handle optional sampling parameters. Map of field name -> (value, Triton dtype)
    optional_inputs = {
        "temperature": (np.float32, "FP32"),
        "top_k": (np.int32, "INT32"),
        "top_p": (np.float32, "FP32"),
        "length_penalty": (np.float32, "FP32"),
        "repetition_penalty": (np.float32, "FP32"),
        "min_length": (np.int32, "INT32"),
        "presence_penalty": (np.float32, "FP32"),
        "frequency_penalty": (np.float32, "FP32"),
        "end_id": (np.int32, "INT32"),
        "pad_id": (np.int32, "INT32"),
        "beam_width": (np.int32, "INT32"),
        "num_return_sequences": (np.int32, "INT32"),
        "random_seed": (np.uint64, "UINT64"),
        "return_log_probs": (np.bool_, "BOOL"),
        "return_context_logits": (np.bool_, "BOOL"),
        "return_generation_logits": (np.bool_, "BOOL"),
        "return_kv_cache_reuse_stats": (np.bool_, "BOOL"),
        "exclude_input_in_output": (np.bool_, "BOOL"),
    }

    for key, (dtype, dtype_str) in optional_inputs.items():
        if key in sampling_parameters:
            val = np.array([[sampling_parameters[key]]], dtype=dtype)
            inp = grpcclient.InferInput(key, val.shape, dtype_str)
            inp.set_data_from_numpy(val)
            inputs.append(inp)

    # Handle optional bad_words and stop_words
    for key in ["bad_words", "stop_words", "embedding_bias_words"]:
        if key in sampling_parameters:
            words = sampling_parameters[key]
            arr = np.array([w.encode("utf-8") for w in words], dtype=np.object_)
            inp = grpcclient.InferInput(key, [len(arr)], "BYTES")
            inp.set_data_from_numpy(arr)
            inputs.append(inp)

    # Handle optional embedding_bias_weights
    if "embedding_bias_weights" in sampling_parameters:
        weights = np.array(sampling_parameters["embedding_bias_weights"], dtype=np.float32)
        inp = grpcclient.InferInput("embedding_bias_weights", list(weights.shape), "FP32")
        inp.set_data_from_numpy(weights)
        inputs.append(inp)

    # Outputs (we just ask for text_output for now)
    outputs = []
    outputs.append(grpcclient.InferRequestedOutput("text_output"))

    request = {
        "model_name": model_name,
        "inputs": inputs,
        "outputs": outputs,
        "request_id": str(request_id),
    }
    # print(f"request: {request}")
    return request


async def main(FLAGS):
    sampling_parameters = {"temperature": "0.01", "top_p": "1.0", "top_k": 20, "min_length": 1}
    max_tokens = 128
    stream = FLAGS.streaming_mode
    model_name = FLAGS.model_name
    with open(FLAGS.input_prompts, "r") as file:
        print(f"Loading inputs from `{FLAGS.input_prompts}`...")
        prompts = file.readlines()

    results_dict = {}
    total_time_sec = 0  # Initialize total time in seconds

    async with grpcclient.InferenceServerClient(url=FLAGS.url, verbose=FLAGS.verbose) as triton_client:
        async def async_request_iterator():
            try:
                for iter in range(FLAGS.iterations):
                    for i, prompt in enumerate(prompts):
                        prompt_id = FLAGS.offset + (len(prompts) * iter) + i
                        results_dict[str(prompt_id)] = []
                        yield create_request(
                            prompt=prompt,
                            stream=stream,
                            request_id=prompt_id,
                            model_name=model_name,
                            max_tokens=max_tokens,
                            sampling_parameters=sampling_parameters,
                        )
            except Exception as error:
                print(f"ERROR: caught error in request iterator: {error}")

        try:
            start_time = time.time()  # Record the start time
            response_iterator = triton_client.stream_infer(
                inputs_iterator=async_request_iterator(),
                stream_timeout=FLAGS.stream_timeout,
            )
            async for response in response_iterator:
                result, error = response
                end_time = time.time()  # Record the end time

                if error:
                    print(f"Encountered error while processing: {error}")
                else:
                    output = result.as_numpy("text_output")
                    # print(f"output: {output[0].decode('utf-8')}")
                    for i in output:
                        debug = {
                            "Prompt": prompts[int(result.get_response().id)],
                            "Response Time": end_time - start_time,
                            "Tokens": count_tokens(i.decode('utf-8')),
                            "Response": i.decode('utf-8'),
                        }
                        results_dict[result.get_response().id] = debug

                    duration = (end_time - start_time)  # Calculate the duration in seconds
                    total_time_sec += (end_time - start_time)  # Add duration to total time in seconds
                    print(f"Model {FLAGS.model_name} - Request {result.get_response().id}: {duration:.2f} seconds")

        except InferenceServerException as error:
            print(error)
            sys.exit(1)

    with open(FLAGS.results_file, "w") as file:
        for key, val in results_dict.items():
            file.write(
                f"Prompt: {val['Prompt']}\nResponse Time: {val['Response Time']}\nTokens: {val['Tokens']}\nResponse: {val['Response']}")
            file.write("\n=========\n\n")
        print(f"Storing results into `{FLAGS.results_file}`...")

    if FLAGS.verbose:
        print(f"\nContents of `{FLAGS.results_file}` ===>")
        system(f"cat {FLAGS.results_file}")

    total_time_ms = total_time_sec  # Convert total time to milliseconds
    print(f"Total time for all requests: {total_time_sec:.2f} seconds ({total_time_ms:.2f} seconds)")
    print("PASS: trtllm example")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8001",
        help="Inference server URL and it gRPC port. Default is localhost:8001.",
    )
    parser.add_argument(
        "-t",
        "--stream-timeout",
        type=float,
        required=False,
        default=None,
        help="Stream timeout in seconds. Default is None.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        required=False,
        default=0,
        help="Add offset to request IDs used",
    )
    parser.add_argument(
        "--input-prompts",
        type=str,
        required=False,
        default="prompts.txt",
        help="Text file with input prompts",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        required=False,
        default="results.txt",
        help="The file with output results",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        required=False,
        default=1,
        help="Number of iterations through the prompts file",
    )
    parser.add_argument(
        "-s",
        "--streaming-mode",
        action="store_true",
        required=False,
        default=False,
        help="Enable streaming mode",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the model to test",
    )
    FLAGS = parser.parse_args()
    asyncio.run(main(FLAGS))
