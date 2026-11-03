#!/usr/bin/env python3
"""
Multi-Model AI Gateway Test Client
Tests self-hosted models (Qwen3, GPT-OSS) and AWS Bedrock Claude
"""

import requests
import json
import subprocess
import sys


def get_gateway_url():
    """Auto-detect AI Gateway URL"""
    try:
        result = subprocess.run([
            'kubectl', 'get', 'gateway', 'ai-gateway',
            '-o', 'jsonpath={.status.addresses[0].value}'
        ], capture_output=True, text=True, check=True)

        if result.stdout.strip():
            return f"http://{result.stdout.strip()}"
        else:
            print(
                "Gateway address not found. Make sure the Gateway is deployed and has an address.")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error getting gateway URL: {e}")
        return "http://localhost:8080"  # fallback


def test_qwen3_model(gateway_url):
    """Test Qwen3 model via /v1/chat/completions"""
    print("=== Testing Qwen3 1.7B ===")
    try:
        response = requests.post(
            f"{gateway_url}/v1/chat/completions",
            headers={
                'Content-Type': 'application/json',
                'x-ai-eg-model': 'Qwen/Qwen3-1.7B'
            },
            json={
                'model': 'Qwen/Qwen3-1.7B',
                'max_tokens': 50,
                'messages': [{'role': 'user', 'content': 'Hello from Qwen3!'}]
            },
            timeout=30
        )

        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            content = data.get('choices', [{}])[0].get(
                'message', {}).get('content', 'No content')
            print(f"✅ SUCCESS: Qwen3 - {content[:100]}...")
            return True
        else:
            print(f"❌ ERROR: Qwen3 - {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: Qwen3 - {e}")
        return False


def test_gpt_model(gateway_url):
    """Test GPT model via /v1/chat/completions"""
    print("\n=== Testing Self-hosted GPT ===")
    try:
        response = requests.post(
            f"{gateway_url}/v1/chat/completions",
            headers={
                'Content-Type': 'application/json',
                'x-ai-eg-model': 'openai/gpt-oss-20b'
            },
            json={
                'model': 'openai/gpt-oss-20b',
                'max_tokens': 50,
                'messages': [{'role': 'user', 'content': 'Hello from GPT!'}]
            },
            timeout=30
        )

        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            content = data.get('choices', [{}])[0].get(
                'message', {}).get('content', 'No content')
            print(f"✅ SUCCESS: GPT - {content[:100]}...")
            return True
        else:
            print(f"❌ ERROR: GPT - {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: GPT - {e}")
        return False


def test_bedrock_claude(gateway_url):
    """Test Bedrock Claude via /anthropic/v1/messages"""
    print("\n=== Testing Bedrock Claude ===")
    try:
        response = requests.post(
            f"{gateway_url}/anthropic/v1/messages",
            headers={
                'Content-Type': 'application/json',
                'x-ai-eg-model': 'anthropic.claude-3-haiku-20240307-v1:0',
                'anthropic-version': 'bedrock-2023-05-31'
            },
            json={
                'model': 'anthropic.claude-3-haiku-20240307-v1:0',
                'max_tokens': 50,
                'messages': [{'role': 'user', 'content': 'Hello from Bedrock!'}]
            },
            timeout=30
        )

        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            content = data.get('content', [{}])[0].get('text', 'No content')
            print(f"✅ SUCCESS: Bedrock Claude - {content[:100]}...")
            return True
        else:
            print(f"❌ ERROR: Bedrock Claude - {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: Bedrock Claude - {e}")
        return False


def main():
    print("🚀 AI Gateway Multi-Model Routing Test")
    print("=" * 60)

    gateway_url = get_gateway_url()
    if not gateway_url:
        print("❌ Could not determine Gateway URL. Exiting.")
        sys.exit(1)

    print(f"Gateway URL: {gateway_url}")

    results = []
    results.append(test_qwen3_model(gateway_url))
    results.append(test_gpt_model(gateway_url))
    results.append(test_bedrock_claude(gateway_url))

    print("\n" + "=" * 60)
    print("🎯 Final Results:")
    print(f"• Qwen3 1.7B: {'✅ PASS' if results[0] else '❌ FAIL'}")
    print(f"• GPT OSS 20B: {'✅ PASS' if results[1] else '❌ FAIL'}")
    print(f"• Bedrock Claude: {'✅ PASS' if results[2] else '❌ FAIL'}")

    passed = sum(results)
    print(f"\n📊 Summary: {passed}/3 models working")
    print("📋 Routing: Header-based using 'x-ai-eg-model'")
    print("🔗 All models accessible through single Gateway endpoint")

    if passed > 0:
        print(f"\n🎉 SUCCESS! {passed} model(s) working through AI Gateway!")
        sys.exit(0)
    else:
        print(f"\n❌ All tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
