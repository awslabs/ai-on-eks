"""
sustained_benchmark.py — Sustained 3-min multi-turn benchmark with Poisson arrivals.

Methodology matched to blog "Cache-aware LLM routing on Amazon EKS":
- 150 users with unique conversation contexts (multi-tenant simulation)
- Poisson arrival pattern (λ=25 QPS)
- Users cycle through turns (context grows each turn)
- Time-series TTFT output bucketed every 30s
- Runs both RR and CA for 3 minutes each with 60s cooldown between strategies

Usage:
  # Discover the cache-aware endpoint first:
  CA_SVC=$(kubectl -n envoy-gateway-system get svc \
    -l gateway.networking.k8s.io/owning-gateway-name=inference-gateway \
    -o jsonpath='{.items[0].metadata.name}')
  CA_EP="http://${CA_SVC}.envoy-gateway-system.svc.cluster.local:8080/v1/completions"

  # Run the benchmark:
  kubectl -n inference exec benchmark-runner -- python3 /tmp/bench.py --ca-endpoint "$CA_EP"
"""
import asyncio, aiohttp, time, json, random, string, os, argparse, sys

random.seed(42)

# === CONFIG ===
MODEL = 'mistralai/Mistral-7B-Instruct-v0.3'
RR = 'http://vllm-inference.inference.svc.cluster.local:8000/v1/completions'

USERS = 150
DURATION_S = 180       # 3 minutes (produces 6 time-series buckets)
BUCKET_S = 30          # time-series bucket width
TARGET_QPS = 25        # Poisson lambda (requests per second)
MAX_TOKENS = 64
MAX_CONCURRENCY = 150
TURN_CONTEXT_TOKENS = 200  # tokens added per turn per user

# ~4500 token system prompt (~12500 chars at ~2.8 chars/token for Mistral)
# Leaves room for per-user context growth within 8192 max_model_len
SYSTEM_PROMPT = (
    "You are a senior AWS Solutions Architect specializing in distributed systems, "
    "Kubernetes, and machine learning infrastructure. You help enterprise customers "
    "design highly available, cost-optimized architectures. "
) + ''.join(random.choices(string.ascii_lowercase + ' ', k=12000))

# Unique per-user context (~60 tokens each, grows per turn)
USER_CONTEXTS = [
    f"CustomerID:{i} Context:" + ''.join(random.choices(string.ascii_lowercase + ' ', k=150))
    for i in range(USERS)
]


def build_prompt(user_id: int, turn: int) -> str:
    """Build prompt with shared prefix + growing per-user context (capped at 10 turns)."""
    effective_turn = min(turn, 10)  # cap to fit 8192 context window
    history = '\n'.join([f"Turn{t}: {USER_CONTEXTS[user_id]}" for t in range(effective_turn + 1)])
    return f"[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n{history}\nProvide a brief answer: [/INST]"


async def send_request(session, endpoint, user_id, turn, sem, results, start_time):
    """Send one request, measure TTFT, record with wall-clock timestamp."""
    async with sem:
        prompt = build_prompt(user_id, turn)
        t0 = time.perf_counter()
        ttft = None
        try:
            async with session.post(
                endpoint,
                json={'model': MODEL, 'prompt': prompt, 'max_tokens': MAX_TOKENS,
                      'stream': True, 'temperature': 0.1},
                timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                if resp.status != 200:
                    results.append({'ts': time.perf_counter() - start_time, 'user': user_id,
                                   'turn': turn, 'ttft': -1})
                    return
                async for line in resp.content:
                    decoded = line.decode()
                    if decoded.strip().startswith('data:') and 'DONE' not in decoded:
                        if not ttft:
                            ttft = (time.perf_counter() - t0) * 1000
                        break
        except Exception:
            results.append({'ts': time.perf_counter() - start_time, 'user': user_id,
                           'turn': turn, 'ttft': -1})
            return

        results.append({'ts': time.perf_counter() - start_time, 'user': user_id,
                       'turn': turn, 'ttft': ttft or 0})


async def run_sustained(name: str, endpoint: str) -> list:
    """Run sustained benchmark with Poisson arrivals for DURATION_S seconds."""
    print(f'\n{"="*60}', flush=True)
    print(f'  {name} — {DURATION_S}s sustained, Poisson \u03bb={TARGET_QPS} QPS', flush=True)
    print(f'{"="*60}', flush=True)

    results = []
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    user_turns = [0] * USERS
    start_time = time.perf_counter()
    tasks = []

    async with aiohttp.ClientSession() as session:
        elapsed = 0.0
        while elapsed < DURATION_S:
            wait = random.expovariate(TARGET_QPS)
            elapsed += wait
            if elapsed >= DURATION_S:
                break

            user_id = random.randint(0, USERS - 1)
            turn = user_turns[user_id]
            user_turns[user_id] += 1

            delay = elapsed - (time.perf_counter() - start_time)
            if delay > 0:
                await asyncio.sleep(delay)

            task = asyncio.create_task(
                send_request(session, endpoint, user_id, turn, sem, results, start_time)
            )
            tasks.append(task)

            if len(tasks) % 100 == 0:
                now = time.perf_counter() - start_time
                pending = len([t for t in tasks if not t.done()])
                print(f'  [{now:.0f}s] sent={len(tasks)} done={len(results)} '
                      f'pending={pending}', flush=True)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # Print time-series summary
    print(f'\n  Time-series (buckets of {BUCKET_S}s):', flush=True)
    print(f'  {"Bucket":>8} | {"p50":>7} | {"p90":>7} | {"n":>4} | {"err":>3}', flush=True)
    print(f'  {"-"*8}-+-{"-"*7}-+-{"-"*7}-+-{"-"*4}-+-{"-"*3}', flush=True)

    num_buckets = DURATION_S // BUCKET_S
    for b in range(num_buckets):
        bucket_data = sorted([r['ttft'] for r in results
                            if r['ttft'] > 0 and int(r['ts'] // BUCKET_S) == b])
        errors = len([r for r in results
                     if r['ttft'] <= 0 and int(r['ts'] // BUCKET_S) == b])
        if bucket_data:
            p50 = bucket_data[len(bucket_data) // 2]
            p90 = bucket_data[int(len(bucket_data) * 0.9)]
            print(f'  {b*BUCKET_S:>3}-{(b+1)*BUCKET_S:>3}s | {p50:>6.0f}ms | {p90:>6.0f}ms | '
                  f'{len(bucket_data):>4} | {errors:>3}', flush=True)
        else:
            print(f'  {b*BUCKET_S:>3}-{(b+1)*BUCKET_S:>3}s | {"N/A":>7} | {"N/A":>7} | '
                  f'{0:>4} | {errors:>3}', flush=True)

    return results


async def warmup(endpoints):
    """Send a few requests to warm up connections."""
    async with aiohttp.ClientSession() as s:
        for ep in endpoints:
            try:
                async with s.post(ep, json={'model': MODEL, 'prompt': '[INST]Hi[/INST]',
                                           'max_tokens': 3},
                                 timeout=aiohttp.ClientTimeout(total=30)) as r:
                    await r.read()
            except Exception:
                pass
    print('Warmup complete.\n', flush=True)


async def main(ca_endpoint: str):
    await warmup([RR, ca_endpoint])

    rr_results = await run_sustained('ROUND-ROBIN', RR)
    print('\n  Cooling down 60s before CA run...', flush=True)
    await asyncio.sleep(60)
    ca_results = await run_sustained('CACHE-AWARE', ca_endpoint)

    # Save raw results
    output = {
        'config': {
            'users': USERS, 'duration_s': DURATION_S, 'target_qps': TARGET_QPS,
            'max_tokens': MAX_TOKENS, 'max_concurrency': MAX_CONCURRENCY,
            'bucket_s': BUCKET_S, 'prefix_tokens': '~4500',
            'arrival_pattern': 'poisson',
            'ca_endpoint': ca_endpoint
        },
        'rr': rr_results,
        'ca': ca_results
    }
    with open('/tmp/sustained_results.json', 'w') as f:
        json.dump(output, f)
    print(f'\nResults saved to /tmp/sustained_results.json', flush=True)

    # Final comparison table
    print(f'\n{"="*60}', flush=True)
    print(f'  COMPARISON: RR vs CA (p50/p90 TTFT per {BUCKET_S}s bucket)', flush=True)
    print(f'{"="*60}', flush=True)
    print(f'  {"Bucket":>8} | {"RR p50":>8} | {"CA p50":>8} | {"RR p90":>8} | {"CA p90":>8} | {"p90 imp":>7}', flush=True)
    print(f'  {"-"*8}-+-{"-"*8}-+-{"-"*8}-+-{"-"*8}-+-{"-"*8}-+-{"-"*7}', flush=True)

    num_buckets = DURATION_S // BUCKET_S
    for b in range(num_buckets):
        rr_b = sorted([r['ttft'] for r in rr_results
                      if r['ttft'] > 0 and int(r['ts'] // BUCKET_S) == b])
        ca_b = sorted([r['ttft'] for r in ca_results
                      if r['ttft'] > 0 and int(r['ts'] // BUCKET_S) == b])
        if rr_b and ca_b:
            rr_p50 = rr_b[len(rr_b) // 2]
            ca_p50 = ca_b[len(ca_b) // 2]
            rr_p90 = rr_b[int(len(rr_b) * 0.9)]
            ca_p90 = ca_b[int(len(ca_b) * 0.9)]
            imp = (rr_p90 - ca_p90) / rr_p90 * 100
            print(f'  {b*BUCKET_S:>3}-{(b+1)*BUCKET_S:>3}s | {rr_p50:>7.0f}ms | '
                  f'{ca_p50:>7.0f}ms | {rr_p90:>7.0f}ms | {ca_p90:>7.0f}ms | {imp:>+5.0f}%', flush=True)

    print(f'\nDone. Copy results: kubectl cp inference/benchmark-runner:/tmp/sustained_results.json ./results/sustained_results.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cache-aware routing sustained benchmark')
    parser.add_argument('--ca-endpoint', type=str,
                        default=os.environ.get('CA_ENDPOINT'),
                        help='Cache-aware (Gateway) endpoint URL. '
                             'Can also be set via CA_ENDPOINT env var.')
    args = parser.parse_args()

    if not args.ca_endpoint:
        print("ERROR: CA endpoint not set. Discover it with:", file=sys.stderr)
        print("  kubectl -n envoy-gateway-system get svc \\", file=sys.stderr)
        print("    -l gateway.networking.k8s.io/owning-gateway-name=inference-gateway \\", file=sys.stderr)
        print("    -o jsonpath='{.items[0].metadata.name}'", file=sys.stderr)
        print("", file=sys.stderr)
        print("Then re-run with:", file=sys.stderr)
        print("  python3 /tmp/bench.py --ca-endpoint http://<name>.envoy-gateway-system.svc.cluster.local:8080/v1/completions", file=sys.stderr)
        sys.exit(1)

    asyncio.run(main(args.ca_endpoint))
