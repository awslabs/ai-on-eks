import os
import argparse
import asyncio
import dataclasses
import json
import csv
import random
import time
import warnings
from dataclasses import dataclass
from typing import AsyncGenerator, List, Optional, Tuple

import numpy as np
import pandas as pd
from backend_request_func import (ASYNC_REQUEST_FUNCS, RequestFuncInput, RequestFuncOutput)
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser

MILLISECONDS_TO_SECONDS_CONVERSION = 1000
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
print(f"setting MODEL_ID={MODEL_ID}")

@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    request_goodput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: List[Tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: List[Tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: List[Tuple[float, float]]
    mean_e2el_ms: float # e2el: end-to-end latency per request
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: List[Tuple[float, float]]


@dataclasses.dataclass
class SampleRequest:
    """A class representing a single inference request for benchmarking.

    Attributes:
        prompt: The input text prompt for the model.
        prompt_len: The length of the prompt in tokens.
        expected_output_len: The expected length of the output in tokens.
    """
    prompt: str
    prompt_len: int
    expected_output_len: int


def sample_requests(
    tokenizer: PreTrainedTokenizerBase,
    args: argparse.Namespace
) -> List[SampleRequest]:
    with open(args.dataset_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    prompts = data.get("prompts", [])

    input_requests = []
    sampled_prompts = random.sample(prompts, args.num_prompts)
    for prompt in sampled_prompts:
        input_requests.append(
            SampleRequest(
                prompt=prompt["text"],
                prompt_len=prompt["token_count"],
                expected_output_len=args.output_len
            )
        )
    return input_requests


async def get_request(
    input_requests: List[SampleRequest],
    request_rate: float,
    burstiness: float = 1.0,
) -> AsyncGenerator[Tuple[int, SampleRequest], None]:
    """
    Asynchronously generates requests at a specified rate
    with OPTIONAL burstiness.

    Args:
        input_requests: A list of input requests, each represented as a tuple.
        request_rate: The rate at which requests are generated (requests/s).
        burstiness (optional): The burstiness factor of the request generation.
            Only takes effect when request_rate is not inf.
            Default value is 1, which follows a Poisson process.
            Otherwise, the request intervals follow a gamma distribution.
            A lower burstiness value (0 < burstiness < 1) results
            in more bursty requests, while a higher burstiness value
            (burstiness > 1) results in a more uniform arrival of requests.
    """
    # calculate scale parameter theta to maintain the desired request_rate.
    assert burstiness > 0, (f"A positive burstiness factor is expected, but given {burstiness}.")

    theta = 1.0 / (request_rate * burstiness)
    input_requests = iter(input_requests)
    for i, request in enumerate(input_requests):
        yield i, request
        # if the request rate is infinity, then we don't need to wait.
        if request_rate == float("inf"):
            continue
        # sample the request interval from the gamma distribution if burstiness is 1, it follows exponential distribution.
        interval = np.random.gamma(shape=burstiness, scale=theta)
        # the next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
    selected_percentile_metrics: List[str],
    selected_percentiles: List[float],
) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens: List[int] = []
    total_input = 0
    completed = 0
    good_completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    all_tpots: List[float] = []
    ttfts: List[float] = []
    e2els: List[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            # We use the tokenizer to count the number of output tokens for all
            # serving backends instead of looking at len(outputs[i].itl) since
            # multiple output tokens may be bundled together
            # Note : this may inflate the output token count slightly
            output_len = len(tokenizer(outputs[i].generated_text, add_special_tokens=False).input_ids)
            actual_output_lens.append(output_len)
            total_input += input_requests[i].prompt_len
            tpot = 0
            if output_len > 1:
                tpot = (outputs[i].latency - outputs[i].ttft) / (output_len - 1)
                tpots.append(tpot)
            outputs[i].tpot = sum(tpots) / len(tpots) if len(tpots) else 0
            # Note: if output_len <= 1, we regard tpot as 0 for goodput
            all_tpots.append(tpot)
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2els.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)

    if completed == 0:
        warnings.warn("All requests failed. This is likely due to a misconfiguration on the benchmark arguments.", stacklevel=2)

    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        request_goodput=good_completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) * 1000,  # Note: ttfts is empty if streaming is not supported by backend
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[(p, np.percentile(ttfts or 0, p) * 1000) for p in selected_percentiles],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[(p, np.percentile(tpots or 0, p) * 1000) for p in selected_percentiles],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[(p, np.percentile(itls or 0, p) * 1000) for p in selected_percentiles],
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[(p, np.percentile(e2els or 0, p) * 1000) for p in selected_percentiles],
    )

    return metrics, actual_output_lens


async def benchmark(
    backend: str,
    api_url: str,
    base_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[SampleRequest],
    request_rate: float,
    burstiness: float,
    disable_tqdm: bool,
    profile: bool,
    selected_percentile_metrics: List[str],
    selected_percentiles: List[str],
    ignore_eos: bool,
    max_concurrency: Optional[int],
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    print("starting initial single prompt test run...")
    test_request = input_requests[0]
    test_input = RequestFuncInput(
        model=model_id,
        prompt=test_request.prompt,
        api_url=api_url,
        prompt_len=test_request.prompt_len,
        output_len=test_request.expected_output_len,
    )
    test_output = await request_func(request_func_input=test_input)
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}")
    else:
        print("Initial test run completed. Starting main benchmark run...")

    if profile:
        print("starting profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            prompt=test_request.prompt,
            api_url=base_url+"/start_profile",
            prompt_len=test_request.prompt_len,
            output_len=test_request.expected_output_len,

        )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("profiler started")

    if burstiness == 1.0:
        distribution = "poisson process"
    else:
        distribution = "gamma distribution"

    print(f"Traffic request rate: {request_rate}")
    print(f"Burstiness factor: {burstiness} ({distribution})")
    print(f"Maximum request concurrency: {max_concurrency}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    # this can be used once the minimum Python version is 3.10 or higher, and it will simplify the code in limited_request_func.
    # semaphore = (asyncio.Semaphore(max_concurrency) if max_concurrency else contextlib.nullcontext())
    semaphore = (asyncio.Semaphore(max_concurrency) if max_concurrency else None)

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await request_func(request_func_input=request_func_input, pbar=pbar)
        async with semaphore:
            return await request_func(request_func_input=request_func_input, pbar=pbar)

    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []
    expected: List[str] = []
    async for i, request in get_request(input_requests, request_rate, burstiness):
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=request.prompt,
            api_url=api_url,
            prompt_len=request.prompt_len,
            output_len=request.expected_output_len,
        )
        tasks.append(asyncio.create_task(limited_request_func(request_func_input=request_func_input, pbar=pbar)))

    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if profile:
        print("stopping profiler...")
        profile_input = RequestFuncInput(
            model=model_id,
            prompt=test_request.prompt,
            api_url=base_url+"/stop_profile",
            prompt_len=test_request.prompt_len,
            output_len=test_request.expected_output_len,
        )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("profiler stopped")

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentile_metrics=selected_percentile_metrics,
        selected_percentiles=selected_percentiles,
    )

    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):", metrics.request_throughput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", metrics.output_throughput))
    print("{:<40} {:<10.2f}".format("Total Token throughput (tok/s):", metrics.total_token_throughput))

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "ttft_description": pd.Series([output.ttft for output in outputs]).describe().to_dict(),
        "tpot_description": pd.Series([output.tpot for output in outputs]).describe().to_dict(),
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "errors": [output.error for output in outputs],
    }

    ret = [{
        'generated': output.generated_text,
        'expected': gt
    } for output, gt in zip(outputs, expected)]

    def process_one_metric(
        metric_attribute_name: str, # E.g., "ttft"
        metric_name: str, # E.g., "TTFT"
        metric_header: str, # E.g., "Time to First Token"
    ):
        # This function prints and adds statistics of the specified metric.
        if metric_attribute_name not in selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c='-'))
        print("{:<40} {:<10.2f}".format(f"Mean {metric_name} (ms):", getattr(metrics, f"mean_{metric_attribute_name}_ms")))
        print("{:<40} {:<10.2f}".format(f"Median {metric_name} (ms):", getattr(metrics, f"median_{metric_attribute_name}_ms")))

        result[f"mean_{metric_attribute_name}_ms"] = getattr(metrics, f"mean_{metric_attribute_name}_ms")
        result[f"median_{metric_attribute_name}_ms"] = getattr(metrics, f"median_{metric_attribute_name}_ms")
        result[f"std_{metric_attribute_name}_ms"] = getattr(metrics, f"std_{metric_attribute_name}_ms")

        for p, value in getattr(metrics, f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):", value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT", "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    print("=" * 50)

    return result, ret


def main(args: argparse.Namespace):
    print("-"*10, "\n", f"args: {args}", "\n", "-"*10)

    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base_url = f"http://{args.host}:{args.port}"

    tokenizer = get_tokenizer(tokenizer_id, trust_remote_code=args.trust_remote_code)

    if args.save_results:
        result_file_name = f"{backend}_{args.request_rate}qps_{args.model.split('/')[-1]}_{args.num_prompts}_out{args.output_len}.txt"
    else:
        result_file_name = None

    print(f"creating input_requests ...")
    input_requests = sample_requests(tokenizer, args)
    print(f"DONE creating input_requests!")
    benchmark_result, ret = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            base_url=base_url,
            model_id=model_id,
            tokenizer=tokenizer,
            input_requests=input_requests,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            disable_tqdm=args.disable_tqdm,
            profile=args.profile,
            selected_percentile_metrics=args.percentile_metrics.split(","),
            selected_percentiles=[float(p) for p in args.metric_percentiles.split(",")],
            ignore_eos=args.ignore_eos,
            max_concurrency=args.max_concurrency,
        ))

    if args.save_results:
        results = {
            "backend": backend,
            "model_id": model_id,
            "tokenizer_id": tokenizer_id,
            "num_prompts": args.num_prompts,
            "request_rate": args.request_rate if args.request_rate < float("inf") else "inf",
            "burstiness": args.burstiness,
            "max_concurrency": args.max_concurrency,
        }
        results = {"outputs": ret, **results, **benchmark_result}

        if args.result_filename:
            result_file_name = args.result_filename
        if args.result_dir:
            result_file_name = os.path.join(args.result_dir, result_file_name)

        # save to file
        # with open(result_file_name, "w", encoding='utf-8') as outfile:
        #     json.dump(results, outfile, indent=4)

        # flat_results = flatten_dict(results)
        file_exists = os.path.isfile(result_file_name)
        # write or append to the CSV file
        with open(result_file_name, mode="a", newline='', encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=results.keys())
            # write header only if the file does not exist
            if not file_exists:
                writer.writeheader()
            # write the row of results
            writer.writerow(results)

        print(f"see results in file: {result_file_name}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="../dataset",
        help="Path to the dataset."
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up.")
    parser.add_argument(
        "--model",
        type=str,
        # required=True,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Name of the model.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Name or path of the tokenizer, if not using the default tokenizer.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    # parser.add_argument(
    #     "--input-len",
    #     type=int,
    #     default=128,
    #     help="Number of input tokens.",
    # )
    parser.add_argument(
        "--output-len",
        type=int,
        default=128,
        help="Number of output tokens.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process or gamma distribution "
        "to synthesize the request arrival times.",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor of the request generation. "
        "Only take effect when request_rate is not inf. "
        "Default value is 1, which follows Poisson process. "
        "Otherwise, the request intervals follow a gamma distribution. "
        "A lower burstiness value (0 < burstiness < 1) results in more "
        "bursty requests. A higher burstiness value (burstiness > 1) "
        "results in a more uniform arrival of requests.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use Torch Profiler. The endpoint must be launched with "
        "VLLM_TORCH_PROFILER_DIR to enable profiler.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="Specify the filename to save benchmark json results."
        "If not specified, results will be saved in "
        "{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
        " format.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Set ignore_eos flag when sending the benchmark request."
        "Warning: ignore_eos is not supported in deepspeed_mii and tgi.")
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default="e2el,ttft,tpot,itl",
        help="Comma-seperated list of selected metrics to report percentils. "
        "This argument specifies the metrics to report percentiles. "
        "Allowed metric names are \"ttft\", \"tpot\", \"itl\", \"e2el\". "
        "Default value is \"ttft,tpot,itl\".")
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="99",
        help="Comma-seperated list of percentiles for selected metrics. "
        "To report 25-th, 50-th, and 75-th percentiles, use \"25,50,75\". "
        "Default value is \"99\". "
        "Use \"--percentile-metrics\" to select metrics.",
    )

    args = parser.parse_args()
    main(args)
