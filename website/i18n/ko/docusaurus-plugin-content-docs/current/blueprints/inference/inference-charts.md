---
sidebar_label: 추론 차트
---

# AI on EKS 추론 차트

AI on EKS 추론 차트는 GPU와 AWS Neuron(Inferentia/Trainium) 하드웨어 모두에서 AI/ML 추론 워크로드를 배포하기 위한 간소화된 Helm 기반 접근 방식을 제공합니다. 이 차트는 여러 배포 구성을 지원하며 인기 있는 모델을 위한 미리 구성된 값이 포함되어 있습니다.

:::info 고급 사용법
자세한 구성 옵션, 고급 배포 시나리오 및 포괄적인 파라미터 문서는 [전체 README](https://github.com/awslabs/ai-on-eks-charts/blob/main/charts/inference-charts/README.md)를 참조하세요.
:::

## 개요

추론 차트는 여러 배포 프레임워크를 지원합니다:

- **VLLM** - 빠른 시작이 가능한 단일 노드 추론
- **Ray-VLLM** - 자동 스케일링 기능이 있는 분산 추론
- **Triton-VLLM** - NVIDIA 추론 서버
- **AIBrix** - AIBrix 전용 구성이 포함된 VLLM
- **LeaderWorkerSet-VLLM** - 대규모 모델을 위한 멀티 노드 추론
- **Diffusers** - 이미지 생성을 위한 Hugging Face Diffusers
- **S3 Model Copy** - Hugging Face에서 S3 스토리지로 모델 다운로드

GPU와 AWS Neuron(Inferentia/Trainium) 가속기 모두 이러한 프레임워크에서 지원됩니다.

## 사전 요구 사항

추론 차트를 배포하기 전에 다음 사항을 확인하세요:

- GPU 또는 AWS Neuron 노드가 있는 Amazon EKS 클러스터([빠른 시작을 위한 추론 준비 클러스터](../../infra/inference/inference-ready-cluster.md))
- Helm 3.0+
- GPU 배포의 경우: NVIDIA 디바이스 플러그인 설치됨
- Neuron 배포의 경우: AWS Neuron 디바이스 플러그인 설치됨
- LeaderWorkerSet 배포의 경우: LeaderWorkerSet CRD 설치됨
- Hugging Face Hub 토큰(`hf-token`이라는 Kubernetes 시크릿으로 저장됨)
- Ray의 경우: KubeRay 인프라
- AIBrix의 경우: AIBrix 인프라
- S3 Model Copy의 경우: S3 쓰기 권한이 있는 서비스 계정

## 빠른 시작

### 1. Hugging Face 토큰 시크릿 생성

[Hugging Face 토큰](https://huggingface.co/docs/hub/en/security-tokens)으로 Kubernetes 시크릿을 생성하세요:

```bash
kubectl create secret generic hf-token --from-literal=token=your_huggingface_token
```

### 2. 미리 구성된 모델 배포

사용 가능한 미리 구성된 모델 중 선택하여 배포하세요:

:::warning

이러한 배포에는 GPU/Neuron 리소스가 필요하며, [활성화](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-resource-limits.html)되어 있어야 하고 CPU 전용 인스턴스보다 비용이 더 많이 듭니다.

:::

```bash
# 차트 저장소 추가
helm repo add ai-on-eks https://awslabs.github.io/ai-on-eks-charts/
helm repo update

# GPU에서 vLLM으로 Qwen 3 1.7B 배포
helm install qwen3-inference ai-on-eks/inference-charts \
  --values https://raw.githubusercontent.com/awslabs/ai-on-eks-charts/refs/heads/main/charts/inference-charts/values-qwen3-1.7b-vllm.yaml

# GPU에서 Ray-vLLM으로 DeepSeek R1 Distill 배포
helm install deepseek-inference ai-on-eks/inference-charts \
  --values https://raw.githubusercontent.com/awslabs/ai-on-eks-charts/refs/heads/main/charts/inference-charts/values-deepseek-r1-distill-llama-8b-ray-vllm-gpu.yaml
```

## 지원 모델

추론 차트에는 다양한 카테고리의 인기 모델을 위한 미리 구성된 값 파일이 포함되어 있습니다:

### 언어 모델

- **DeepSeek R1 Distill Llama 8B** - 고급 추론 모델
- **Llama 3.2 1B** - 경량 언어 모델
- **Llama 4 Scout 17B** - 중간 크기 언어 모델
- **Mistral Small 24B** - 효율적인 대규모 언어 모델
- **GPT OSS 20B** - 오픈소스 GPT 변형
- **Qwen3 1.7B** - 컴팩트한 다국어 언어 모델

### Diffusion 모델

- **FLUX.1 Schnell** - 빠른 텍스트-이미지 생성
- **Stable Diffusion XL** - 고품질 이미지 생성
- **Stable Diffusion 3.5** - 향상된 기능이 있는 최신 SD 모델
- **Kolors** - 예술적 이미지 생성
- **OmniGen** - 멀티모달 생성

### Neuron 최적화 모델

- **Llama 2 13B** - AWS Inferentia에 최적화됨
- **Llama 3 70B** - Inferentia에서의 대규모 모델
- **Llama 3.1 8B** - 효율적인 Inferentia 배포

각 모델에는 다양한 프레임워크(VLLM, Ray-VLLM, Triton-VLLM 등)를 위한 최적화된 구성이 포함되어 있습니다.

## 배포 예제

### 언어 모델 배포

```bash
# 차트 저장소 추가
helm repo add ai-on-eks https://awslabs.github.io/ai-on-eks-charts/
helm repo update

# GPU에서 vLLM으로 Qwen 3 1.7B 배포
helm install qwen3-inference ai-on-eks/inference-charts \
  --values https://raw.githubusercontent.com/awslabs/ai-on-eks-charts/refs/heads/main/charts/inference-charts/values-qwen3-1.7b-vllm.yaml

# GPU에서 Ray-vLLM으로 DeepSeek R1 Distill 배포
helm install deepseek-inference ai-on-eks/inference-charts \
  --values https://raw.githubusercontent.com/awslabs/ai-on-eks-charts/refs/heads/main/charts/inference-charts/values-deepseek-r1-distill-llama-8b-ray-vllm-gpu.yaml

# LeaderWorkerSet-VLLM으로 Llama 4 Scout 17B 배포
helm install llama4-lws ai-on-eks/inference-charts \
  --values https://raw.githubusercontent.com/awslabs/ai-on-eks-charts/refs/heads/main/charts/inference-charts/values-llama-4-scout-17b-lws-vllm.yaml
```

### Diffusion 모델 배포

```bash
# 이미지 생성을 위한 FLUX.1 Schnell 배포
helm install flux-diffusers ai-on-eks/inference-charts \
  --values https://raw.githubusercontent.com/awslabs/ai-on-eks-charts/refs/heads/main/charts/inference-charts/values-flux-1-diffusers.yaml

# Stable Diffusion XL 배포
helm install sdxl-diffusers ai-on-eks/inference-charts \
  --values https://raw.githubusercontent.com/awslabs/ai-on-eks-charts/refs/heads/main/charts/inference-charts/values-stable-diffusion-xl-base-1-diffusers.yaml
```

### Neuron 배포

```bash
# Inferentia에서 Llama 3.1 8B 배포
helm install llama31-neuron ai-on-eks/inference-charts \
  --values https://raw.githubusercontent.com/awslabs/ai-on-eks-charts/refs/heads/main/charts/inference-charts/values-llama-31-8b-vllm-neuron.yaml

# Inferentia에서 Ray-VLLM으로 Llama 3 70B 배포
helm install llama3-70b-neuron ai-on-eks/inference-charts \
  --values https://raw.githubusercontent.com/awslabs/ai-on-eks-charts/refs/heads/main/charts/inference-charts/values-llama-3-70b-ray-vllm-neuron.yaml
```

### S3 Model Copy

S3 Model Copy 기능을 사용하면 Hugging Face Hub에서 모델을 다운로드하여 S3 스토리지에 업로드할 수 있습니다. 이는 다음과 같은 경우에 유용합니다:

- 더 빠른 배포를 위해 S3에 모델 사전 준비
- 프라이빗 S3 버킷에 모델 저장소 생성
- AWS 내부 네트워크를 활용하여 추론 시작 시간 단축

```bash
# Hugging Face에서 S3로 Llama 3 8B 모델 복사
helm install s3-copy-llama3 ai-on-eks/inference-charts \
  --values https://raw.githubusercontent.com/awslabs/ai-on-eks-charts/refs/heads/main/charts/inference-charts/values-s3-copy-llama3-8b.yaml
```

#### 커스텀 S3 Model Copy

S3에 모든 모델을 복사하기 위한 커스텀 값 파일을 생성하세요:

```yaml
s3ModelCopy:
  namespace: default
  model: deepseek-ai/DeepSeek-R1
  s3Path: my-models-bucket/ # 모델은 s3://my-models-bucket/deepseek-ai/DeepSeek-R1로 복사됩니다

serviceAccountName: s3-copy-service-account  # S3 쓰기 권한이 있는 서비스 계정
```

S3 복사 작업 배포:

```bash
helm install custom-s3-copy ai-on-eks/inference-charts \
  --values custom-s3-copy-values.yaml
```

:::info S3 권한
서비스 계정에는 대상 S3 버킷에 쓰기 위한 IAM 권한이 필요합니다. 서비스 계정에 S3 권한을 부여하려면 [Pod Identity](https://docs.aws.amazon.com/eks/latest/userguide/pod-identities.html) 사용을 고려하세요.
:::

## 구성

### 주요 파라미터

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `inference.accelerator` | 가속기 유형(`gpu` 또는 `neuron`) | `gpu` |
| `inference.framework` | 프레임워크(`vllm`, `ray-vllm`, `triton-vllm`, `aibrix` 등) | `vllm` |
| `inference.serviceName` | 추론 서비스 이름 | `inference` |
| `inference.modelServer.deployment.replicas` | 레플리카 수 | `1` |
| `model` | Hugging Face Hub의 모델 ID | `NousResearch/Llama-3.2-1B` |
| `modelParameters.gpuMemoryUtilization` | GPU 메모리 활용도 | `0.8` |
| `modelParameters.maxModelLen` | 최대 모델 시퀀스 길이 | `8192` |
| `modelParameters.tensorParallelSize` | 텐서 병렬 크기 | `1` |
| `modelParameters.pipelineParallelSize` | 파이프라인 병렬 크기 | `1` |
| `s3ModelCopy.namespace` | S3 모델 복사 작업의 네임스페이스 | `default` |
| `s3ModelCopy.model` | S3에 복사할 Hugging Face 모델 ID | 설정 안 됨 |
| `s3ModelCopy.s3Path` | 모델을 업로드할 S3 경로 | 설정 안 됨 |
| `serviceAccountName` | 서비스 계정 이름 | `default` |

### 커스텀 구성

커스텀 값 파일을 생성하세요:

```yaml
inference:
  accelerator: gpu  # 또는 neuron
  framework: vllm   # vllm, ray-vllm, triton-vllm, aibrix, lws-vllm, diffusers
  serviceName: my-inference
  modelServer:
    deployment:
      replicas: 1
      instanceType: g5.2xlarge

model: "NousResearch/Llama-3.2-1B"
modelParameters:
  gpuMemoryUtilization: 0.8
  maxModelLen: 8192
  tensorParallelSize: 1
```

커스텀 값으로 배포:

```bash
helm install my-inference ai-on-eks/inference-charts \
  --values custom-values.yaml
```

## API 사용법

배포된 서비스는 프레임워크에 따라 다른 API 엔드포인트를 노출합니다:

### VLLM/Ray-VLLM

- `/v1/models` - 사용 가능한 모델 목록
- `/v1/chat/completions` - 채팅 완성 API
- `/v1/completions` - 텍스트 완성 API
- `/metrics` - Prometheus 메트릭

### Triton-VLLM

- `/v2/models` - 사용 가능한 모델 목록
- `/v2/models/vllm_model/generate` - 모델 추론
- `/v2/health/ready` - 헬스 체크

### Diffusers

- `/v1/generations` - 이미지 생성 API

### 사용 예제

포트 포워딩으로 서비스에 접근:

```bash
kubectl port-forward svc/<service-name> 8000
```

API 테스트:

```bash
# 채팅 완성 (VLLM/Ray-VLLM)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

# 이미지 생성 (Diffusers)
curl -X POST http://localhost:8000/v1/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "A beautiful sunset over mountains"}'
```

## 문제 해결

### 일반적인 문제

1. **파드가 Pending 상태에서 멈춤**
    - GPU/Neuron 노드가 사용 가능한지 확인
    - 리소스 요청이 사용 가능한 하드웨어와 일치하는지 확인
    - LeaderWorkerSet 배포의 경우: LeaderWorkerSet CRD가 설치되어 있는지 확인

2. **모델 다운로드 실패**
    - Hugging Face 토큰이 `hf-token` 시크릿으로 올바르게 구성되어 있는지 확인
    - Hugging Face Hub에 대한 네트워크 연결 확인
    - 모델 ID가 정확하고 접근 가능한지 확인

3. **메모리 부족 오류**
    - `gpuMemoryUtilization` 파라미터 조정(0.8에서 0.7로 줄여보기)
    - 더 큰 모델의 경우 텐서 병렬화 사용 고려
    - 대규모 모델의 경우 여러 GPU를 사용하는 LeaderWorkerSet 또는 Ray 배포 사용

4. **Ray 배포 문제**
    - KubeRay 인프라가 설치되어 있는지 확인
    - Ray 클러스터 상태 및 워커 연결 확인
    - Ray 버전 호환성 확인

5. **Triton 배포 문제**
    - Triton 서버 로그에서 모델 로딩 오류 확인
    - 모델 저장소 구성 확인
    - 적절한 헬스 체크 엔드포인트에 접근 가능한지 확인

### 로그

프레임워크에 따른 배포 로그 확인:

### 로그 확인

```bash
# VLLM 배포
kubectl logs -l app.kubernetes.io/component=<service-name>

# Ray 배포
kubectl logs -l ray.io/node-type=head
kubectl logs -l ray.io/node-type=worker

# LeaderWorkerSet 배포
kubectl logs -l leaderworkerset.sigs.k8s.io/role=leader
```

## 다음 단계

- GPU 배포를 위한 [GPU 전용 구성](/docs/category/gpu-inference-on-eks) 살펴보기
- Inferentia 배포를 위한 [Neuron 전용 구성](/docs/category/neuron-inference-on-eks) 알아보기
