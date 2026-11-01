#!/usr/bin/env python3
"""
Test script for token-based rate limiting with GPT and Bedrock Claude
"""

import requests
import json
import time
import subprocess


def get_gateway_url():
    """Auto-detect the Gateway URL using kubectl"""
    try:
        result = subprocess.run([
            'kubectl', 'get', 'gateway', 'ai-gateway',
            '-o', 'jsonpath={.status.addresses[0].value}'
        ], capture_output=True, text=True, check=True)

        if result.stdout.strip():
            return f"http://{result.stdout.strip()}"
        else:
            print("Gateway address not found.")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error getting gateway URL: {e}")
        return None


def test_gpt_rate_limiting(gateway_url):
    """Test rate limiting with GPT model"""
    print("🔹 Testing GPT Model Rate Limiting")
    print("-" * 40)

    model_name = "openai/gpt-oss-20b"
    user_id = "test-user-gpt"

    headers = {
        "Content-Type": "application/json",
        "x-ai-eg-model": model_name,
        "x-user-id": user_id
    }

    payload = {
        "model": model_name,
        "prompt": "Hello! This is a rate limiting test.",
        "max_tokens": 5,
        "temperature": 0.7
    }

    successful_requests = 0
    rate_limited_requests = 0

    # Send multiple requests to trigger rate limiting
    for i in range(1, 6):  # Test with 5 requests
        try:
            response = requests.post(
                f"{gateway_url}/v1/completions",
                headers=headers,
                json=payload,
                timeout=10
            )

            print(f"Request {i}: Status {response.status_code}", end="")

            if response.status_code == 200:
                successful_requests += 1
                result = response.json()
                tokens = result.get('usage', {}).get('total_tokens', 'N/A')
                print(f" ✅ SUCCESS - Tokens: {tokens}")
            elif response.status_code == 429:
                rate_limited_requests += 1
                print(f" 🚫 RATE LIMITED")
            else:
                print(f" ❌ ERROR")

        except requests.exceptions.RequestException as e:
            print(f"Request {i}: ❌ Request failed")

        time.sleep(0.5)

    return successful_requests, rate_limited_requests


def test_bedrock_rate_limiting(gateway_url):
    """Test rate limiting with Bedrock Claude"""
    print("\n🔹 Testing Bedrock Claude Rate Limiting")
    print("-" * 40)

    model_name = "anthropic.claude-3-haiku-20240307-v1:0"
    user_id = "test-user-bedrock"

    headers = {
        "Content-Type": "application/json",
        "x-ai-eg-model": model_name,
        "x-user-id": user_id
    }

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": "Hello! This is a rate limiting test."
            }
        ],
        "max_tokens": 5,
        "anthropic_version": "bedrock-2023-05-31"
    }

    successful_requests = 0
    rate_limited_requests = 0

    # Send multiple requests to trigger rate limiting
    for i in range(1, 6):  # Test with 5 requests
        try:
            response = requests.post(
                f"{gateway_url}/anthropic/v1/messages",
                headers=headers,
                json=payload,
                timeout=10
            )

            print(f"Request {i}: Status {response.status_code}", end="")

            if response.status_code == 200:
                successful_requests += 1
                result = response.json()
                usage = result.get('usage', {})
                input_tokens = usage.get('input_tokens', 0)
                output_tokens = usage.get('output_tokens', 0)
                total_tokens = input_tokens + output_tokens
                print(f" ✅ SUCCESS - Tokens: {total_tokens}")
            elif response.status_code == 429:
                rate_limited_requests += 1
                print(f" 🚫 RATE LIMITED")
            else:
                print(f" ❌ ERROR")

        except requests.exceptions.RequestException as e:
            print(f"Request {i}: ❌ Request failed")

        time.sleep(0.5)

    return successful_requests, rate_limited_requests


def main():
    """Main test function"""
    gateway_url = get_gateway_url()
    if not gateway_url:
        print("❌ Could not determine Gateway URL. Exiting.")
        return

    print("🚀 Testing Token-Based Rate Limiting")
    print("=" * 50)
    print(f"Gateway URL: {gateway_url}")

    # Test GPT rate limiting
    gpt_success, gpt_limited = test_gpt_rate_limiting(gateway_url)

    # Test Bedrock rate limiting
    bedrock_success, bedrock_limited = test_bedrock_rate_limiting(gateway_url)

    # Summary
    print("\n" + "=" * 50)
    print("FINAL SUMMARY:")
    print(f"GPT Model: {gpt_success} successful, {gpt_limited} rate limited")
    print(f"Bedrock Claude: {bedrock_success} successful, {
          bedrock_limited} rate limited")

    if gpt_limited > 0 or bedrock_limited > 0:
        print("✅ Rate limiting is working!")
    elif gpt_success > 0 or bedrock_success > 0:
        print("⚠️  Rate limiting may not be active yet")
    else:
        print("❌ No successful requests - check configuration")


if __name__ == "__main__":
    main()
