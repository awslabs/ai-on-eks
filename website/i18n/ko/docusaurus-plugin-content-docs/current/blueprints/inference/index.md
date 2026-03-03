---
sidebar_position: 1
---

# EKS에서의 추론

Amazon EKS에서 대규모 언어 모델(LLM) 및 기타 AI 모델을 배포하고 실행하세요.

## 이 섹션의 내용

이 섹션에서는 EKS에서 추론 워크로드를 실행하기 위한 실용적인 배포 가이드와 Helm 차트를 제공합니다. 오픈소스 LLM, Diffusion 모델 또는 커스텀 AI 모델을 배포하든, 즉시 사용 가능한 구성과 단계별 지침을 찾을 수 있습니다.

---

## [추론 차트](./inference-charts.md)

최적의 성능을 위한 사전 구성된 값과 함께 EKS에서 인기 있는 AI 모델을 배포하기 위한 Helm 차트입니다.

**제공 내용:**
- vLLM, Ray-vLLM, Triton, Diffusers를 위한 즉시 배포 가능한 Helm 차트
- 인기 모델(Llama, DeepSeek, Mistral, Stable Diffusion 등)을 위한 사전 구성된 values 파일
- GPU(NVIDIA) 및 Neuron(AWS Inferentia/Trainium) 배포 모두 지원
- 헬스 체크, 오토스케일링, 모니터링이 포함된 구성

**사용 사례:**
- 오픈소스 LLM의 빠른 배포
- 조직 전체의 표준화된 배포 패턴
- 커스텀 모델 배포를 위한 참조 구현

[추론 차트 살펴보기 →](./inference-charts.md)

---

## 프레임워크별 배포 가이드

EKS에서 특정 프레임워크를 사용한 모델 배포에 대한 상세 가이드로, 하드웨어 유형별로 구성되어 있습니다.

### GPU 배포

NVIDIA GPU에서 모델을 배포하기 위한 단계별 가이드:

- **[AIBrix DeepSeek Distill](./framework-guides/GPUs/aibrix-deepseek-distill.md)** - AIBrix 최적화로 DeepSeek R1 Distill Llama 8B 배포
- **[NVIDIA Dynamo](./framework-guides/GPUs/nvidia-dynamo.md)** - NVIDIA Dynamo 프레임워크로 모델 배포
- **[NVIDIA NIM Llama 3](./framework-guides/GPUs/nvidia-nim-llama3.md)** - NVIDIA NIM을 사용하여 Llama 3 배포
- **[NVIDIA NIM Operator](./framework-guides/GPUs/nvidia-nim-operator.md)** - NVIDIA NIM 배포를 위한 Kubernetes 오퍼레이터
- **[vLLM과 NVIDIA Triton Server](./framework-guides/GPUs/vLLM-NVIDIATritonServer.md)** - Triton과 vLLM을 사용한 추론
- **[vLLM과 Ray Serve](./framework-guides/GPUs/vLLM-rayserve.md)** - Ray Serve와 vLLM을 사용한 확장 가능한 추론

### Neuron 배포

AWS Inferentia 및 Trainium에서 모델을 배포하기 위한 단계별 가이드:

- **[Inferentia2에서의 Mistral 7B](./framework-guides/Neuron/Mistral-7b-inf2.md)** - AWS Inferentia 2에서 Mistral 7B 배포
- **[Inferentia2에서의 Llama 2](./framework-guides/Neuron/llama2-inf2.md)** - AWS Inferentia 2에서 Llama 2 13B 배포
- **[Inferentia2에서의 Llama 3](./framework-guides/Neuron/llama3-inf2.md)** - AWS Inferentia 2에서 Llama 3 배포
- **[Ray Serve 고가용성](./framework-guides/Neuron/rayserve-ha.md)** - Neuron에서 고가용성 Ray Serve 배포
- **[Inferentia2에서의 Stable Diffusion](./framework-guides/Neuron/stablediffusion-inf2.md)** - AWS Inferentia 2에서 Stable Diffusion 배포
- **[Inferentia2에서의 vLLM Ray](./framework-guides/Neuron/vllm-ray-inf2.md)** - AWS Inferentia 2에서 Ray와 vLLM 배포



---

## 시작하기

1. **인프라 설정** - AI/ML 워크로드에 최적화된 EKS 클러스터를 프로비저닝하기 위해 [추론 준비 클러스터](../../infra/inference/inference-ready-cluster.md)로 시작하세요

2. **배포 방법 선택**:
   - 인기 모델의 빠른 배포 → [추론 차트](./inference-charts.md) 사용
   - 특정 프레임워크 또는 커스텀 구성 → 위의 프레임워크별 가이드 참조

3. **배포 최적화** - [가이던스 섹션](../../guidance/index.md)의 모범 사례를 적용하여 성능을 개선하고 비용을 절감하세요

---

## 도움이 필요하신가요?

- **인프라 설정**: 클러스터 설정 및 구성은 [추론 인프라](../../infra/inference/index.md) 참조
- **최적화**: 성능 튜닝 및 모범 사례는 [가이던스 섹션](../../guidance/index.md) 확인
- **이슈**: [GitHub Issues](https://github.com/awslabs/ai-on-eks/issues)에서 버그 리포트 또는 기능 요청
- **커뮤니티**: [GitHub Discussions](https://github.com/awslabs/ai-on-eks/discussions)에서 토론 참여
