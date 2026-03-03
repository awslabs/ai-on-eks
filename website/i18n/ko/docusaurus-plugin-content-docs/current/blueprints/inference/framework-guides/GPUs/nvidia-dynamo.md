---
title: Amazon EKS의 NVIDIA Dynamo
sidebar_position: 8
---

import CollapsibleContent from '@site/src/components/CollapsibleContent';

:::warning
EKS에 ML 모델을 배포하려면 GPU 또는 Neuron 인스턴스에 대한 액세스가 필요합니다. 배포가 작동하지 않는 경우 이러한 리소스에 대한 액세스가 누락되어 있기 때문인 경우가 많습니다. 또한 일부 배포 패턴은 Karpenter 오토스케일링 및 정적 노드 그룹에 의존합니다. 노드가 초기화되지 않으면 Karpenter 또는 노드 그룹의 로그를 확인하여 문제를 해결하십시오.
:::

:::info
NVIDIA Dynamo는 대규모로 AI 추론(Inference) 그래프를 배포하고 관리하기 위한 클라우드 네이티브 플랫폼입니다. 이 구현은 Amazon EKS에서 엔터프라이즈급 모니터링과 확장성을 갖춘 완전한 인프라 설정을 제공합니다.
:::

# Amazon EKS의 NVIDIA Dynamo

:::warning 활발한 개발 중
이 NVIDIA Dynamo 블루프린트는 현재 **활발한 개발 중**입니다. 사용자 경험과 기능을 지속적으로 개선하고 있습니다. 사용자 피드백과 모범 사례를 기반으로 구현을 반복하고 향상시킴에 따라 기능, 구성 및 배포 프로세스가 릴리스 간에 변경될 수 있습니다.

향후 릴리스에서 반복적인 개선이 있을 것으로 예상됩니다. 문제가 발생하거나 개선 제안이 있으면 이슈를 열거나 프로젝트에 기여해 주십시오.
:::

## 빠른 시작

**즉시 시작하고 싶으신가요?** 최소한의 명령 시퀀스입니다:

```bash
# 1. 클론 및 이동
git clone https://github.com/awslabs/ai-on-eks.git && cd ai-on-eks/infra/nvidia-dynamo

# 2. 인프라 및 플랫폼 배포 (15-30분)
./install.sh

# 3. 사전 빌드된 NGC 컨테이너를 사용하여 추론 예제 배포
cd ../../blueprints/inference/nvidia-dynamo

./deploy.sh                # 예제를 선택하는 대화형 메뉴
# ./deploy.sh vllm           # 대화형 설정으로 vLLM 배포

# 4. 배포 테스트 (모델 다운로드 대기)
kubectl port-forward svc/vllm-frontend 8000:8000 -n dynamo-cloud
curl http://localhost:8000/health
```

**사전 요구 사항**: AWS CLI, kubectl, helm, terraform, git, NGC API 토큰, HuggingFace 토큰 ([아래 자세한 설정](#사전-요구-사항))

---

## NVIDIA Dynamo란?

[NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo)는 대규모 언어 모델(LLM) 및 생성형 AI 애플리케이션의 성능과 확장성을 최적화하도록 설계된 오픈 소스 추론 프레임워크입니다. Apache 2.0 라이선스 하에 릴리스된 Dynamo는 여러 GPU와 노드에 걸쳐 복잡한 AI 워크로드를 오케스트레이션하는 데이터센터 규모의 분산 추론 서빙 프레임워크를 제공합니다.

### 추론 그래프란?

**추론 그래프(Inference Graph)**는 상호 연결된 노드를 통해 AI 모델이 데이터를 처리하는 방식을 정의하는 계산 워크플로우로, 다음과 같은 복잡한 다단계 AI 작업을 가능하게 합니다:
- **LLM 체인**: 여러 언어 모델을 통한 순차적 처리
- **멀티모달 처리**: 텍스트, 이미지 및 오디오 처리 결합
- **사용자 정의 추론 파이프라인**: 특정 AI 애플리케이션을 위한 맞춤형 워크플로우
- **분리된 서빙**: 최적의 리소스 활용을 위해 prefill과 decode 단계 분리

## 개요

이 블루프린트는 [NVIDIA NGC 카탈로그](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo)의 **[공식 NVIDIA Dynamo Helm 차트](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/helm-charts/dynamo-platform)**를 사용하며, Amazon EKS에서의 배포 프로세스를 단순화하기 위한 추가 셸 스크립트와 Terraform 자동화를 제공합니다.

### 배포 접근 방식

**이 설정 프로세스의 이유는?**
이 구현은 여러 단계를 포함하지만 간단한 Helm 전용 배포에 비해 여러 가지 이점을 제공합니다:

- **완전한 인프라**: VPC, EKS 클러스터, ECR 리포지토리 및 모니터링 스택을 자동으로 프로비저닝
- **프로덕션 준비**: 엔터프라이즈급 보안, 모니터링 및 확장성 기능 포함
- **AWS 통합**: EKS 오토스케일링, EFA 네트워킹 및 AWS 서비스 활용
- **사용자 정의 가능**: GPU 노드 풀, 네트워킹 및 리소스 할당의 미세 조정 가능
- **재현 가능**: Infrastructure as Code로 환경 전반에 걸쳐 일관된 배포 보장

**더 간단한 배포의 경우**: EKS 클러스터가 이미 있고 최소한의 설정을 선호하는 경우 소스 저장소에서 직접 Dynamo Helm 차트를 사용할 수 있습니다. 이 블루프린트는 완전한 프로덕션 준비 경험을 제공합니다.

LLM 및 생성형 AI 애플리케이션이 점점 보편화됨에 따라 효율적이고 확장 가능하며 저지연 추론 솔루션에 대한 수요가 증가했습니다. 기존 추론 시스템은 특히 분산 다중 노드 환경에서 이러한 요구를 충족하는 데 어려움을 겪는 경우가 많습니다. NVIDIA Dynamo는 Amazon S3, Elastic Fabric Adapter (EFA) 및 Amazon EKS와 같은 AWS 서비스를 지원하여 성능과 확장성을 최적화하는 혁신적인 솔루션을 제공함으로써 이러한 문제를 해결합니다.

### 주요 기능

**성능 최적화:**
- **분리된 서빙**: 최적의 리소스 활용을 위해 다른 GPU에서 prefill과 decode 단계 분리
- **동적 GPU 스케줄링**: NVIDIA Dynamo Planner를 통한 실시간 수요 기반 지능형 리소스 할당
- **스마트 요청 라우팅**: 관련 캐시된 데이터가 있는 워커로 요청을 라우팅하여 KV 캐시 재계산 최소화
- **가속화된 데이터 전송**: NVIDIA NIXL 라이브러리를 통한 저지연 통신
- **효율적인 KV 캐시 관리**: KV Cache Block Manager를 통한 메모리 계층 전반의 지능형 오프로딩

**인프라 준비:**
- **추론 엔진 불가지론**: TensorRT-LLM, vLLM, SGLang 및 기타 런타임 지원
- **모듈식 설계**: 기존 AI 스택에 맞는 구성 요소 선택
- **엔터프라이즈급**: 완전한 모니터링, 로깅 및 보안 통합
- **Amazon EKS 최적화**: EKS 오토스케일링, GPU 지원 및 AWS 서비스 활용

## 아키텍처

배포는 다음 구성 요소와 함께 Amazon EKS를 사용합니다:

![NVIDIA Dynamo 아키텍처](https://github.com/ai-dynamo/dynamo/blob/main/docs/images/architecture.png?raw=true)

**주요 구성 요소:**
- **VPC 및 네트워킹**: 저지연 노드 간 통신을 위한 EFA 지원이 있는 표준 VPC
- **EKS 클러스터**: Karpenter를 사용하는 GPU 지원 노드 그룹이 있는 관리형 Kubernetes
- **Dynamo Platform**: Operator, API Store 및 지원 서비스 (NATS, PostgreSQL, MinIO)
- **모니터링 스택**: Prometheus, Grafana 및 AI/ML 관측성
- **스토리지**: 공유 모델 스토리지 및 캐싱을 위한 Amazon EFS

## 사전 요구 사항

**시스템 요구 사항**: Ubuntu 22.04 또는 24.04 (NVIDIA Dynamo는 공식적으로 이러한 버전만 지원)

설정 호스트에 다음 도구를 설치하십시오 (권장: EKS 및 ECR 권한이 있는 t3.xlarge 이상의 EC2 인스턴스):

- **AWS CLI**: 적절한 권한으로 구성됨 ([설치 가이드](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html))
- **kubectl**: Kubernetes 명령줄 도구 ([설치 가이드](https://kubernetes.io/docs/tasks/tools/install-kubectl/))
- **helm**: Kubernetes 패키지 관리자 ([설치 가이드](https://helm.sh/docs/intro/install/))
- **terraform**: Infrastructure as code 도구 ([설치 가이드](https://learn.hashicorp.com/tutorials/terraform/install-cli))
- **git**: 버전 제어 ([설치 가이드](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git))
- **Python 3.10+**: pip 및 venv 포함 ([설치 가이드](https://www.python.org/downloads/))
- **EKS 클러스터**: 버전 1.33 (테스트 및 지원됨)

### 필수 API 토큰

- **[NGC API 토큰](https://catalog.ngc.nvidia.com/)**: NVIDIA의 사전 빌드된 Dynamo 컨테이너 이미지에 액세스하는 데 필요
  - [NVIDIA NGC](https://catalog.ngc.nvidia.com/)에 가입
  - 계정 설정에서 API 키 생성
  - `NGC_API_KEY` 환경 변수로 설정하거나 설치 중에 제공
- **[HuggingFace 토큰](https://huggingface.co/settings/tokens)**: 모델 다운로드에 필요
  - [HuggingFace](https://huggingface.co/)에서 계정 생성
  - 모델 읽기 권한이 있는 액세스 토큰 생성
  - `HF_TOKEN` 환경 변수로 설정하거나 배포 중 대화형으로 제공

<CollapsibleContent header={<h2><span>솔루션 배포</span></h2>}>

Amazon EKS에 NVIDIA Dynamo를 배포하려면 다음 단계를 완료하십시오:

<a id="1단계-저장소-클론"></a>
### 1단계: 저장소 클론

```bash
git clone https://github.com/awslabs/ai-on-eks.git && cd ai-on-eks
```

<a id="2단계-인프라-및-플랫폼-배포"></a>
### 2단계: 인프라 및 플랫폼 배포

인프라 디렉토리로 이동하고 설치 스크립트를 실행합니다:

```bash
cd infra/nvidia-dynamo
./install.sh
```

이 명령은 완전한 환경을 프로비저닝합니다:
- **VPC**: 서브넷, 보안 그룹, NAT 게이트웨이 및 인터넷 게이트웨이
- **EKS 클러스터**: Karpenter를 사용하는 GPU 지원 노드 그룹 포함
- **모니터링 스택**: Prometheus, Grafana 및 AI/ML 관측성
- **ArgoCD**: GitOps 배포 플랫폼
- **Dynamo Platform**: [공식 NVIDIA Dynamo Helm 차트](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/helm-charts/dynamo-platform)를 사용하여 배포 (Operator, API Store, NATS, PostgreSQL, MinIO)

**소요 시간**: 15-30분

<a id="3단계-추론-예제-배포"></a>
### 3단계: 추론 예제 배포

사전 빌드된 [NGC 컨테이너 이미지](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers)를 사용하여 간소화된 배포 스크립트로 추론 서비스를 배포합니다:

```bash
cd ../../blueprints/inference/nvidia-dynamo

# 9개 예제 중 선택하는 대화형 메뉴
./deploy.sh

# 또는 특정 예제를 직접 배포
./deploy.sh vllm           # vLLM 집계 서빙
./deploy.sh sglang         # RadixAttention이 있는 SGLang
./deploy.sh hello-world    # CPU 전용 테스트
./deploy.sh trtllm         # TensorRT-LLM 최적화
```

**사용 가능한 예제:**
- **hello-world**: CPU 전용 연결 테스트
- **vllm**: OpenAI API가 있는 vLLM 집계 서빙
- **sglang**: 고급 RadixAttention 캐싱이 있는 SGLang
- **trtllm**: TensorRT-LLM 최적화 추론
- **multi-replica-vllm**: KV 라우팅 및 고가용성이 있는 다중 레플리카 배포
- **vllm-disagg**: 분리된 prefill/decode 워커
- **sglang-disagg**: RadixAttention이 있는 SGLang 분리 서빙
- **trtllm-disagg**: TensorRT-LLM 분리 서빙
- **kv-routing**: KV 인식 지능형 라우팅

**사전 빌드된 컨테이너의 주요 이점:**
- **빌드 불필요**: 공식 [NGC 컨테이너 이미지](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo) (v0.4.1) 사용
- **더 빠른 배포**: 20분 이상의 빌드 프로세스 생략
- **일관된 경험**: NVIDIA에서 테스트 및 검증된 이미지
- **버전 관리**: `blueprint.tfvars`에서 자동 버전 감지
- **재정의 지원**: `DYNAMO_VERSION=v0.4.1 ./deploy.sh`를 사용하여 버전 재정의

</CollapsibleContent>

## 사용 가능한 예제

### 프로덕션 준비 예제

다음 예제는 완전히 테스트되었고 포괄적인 문서와 함께 프로덕션 준비가 되어 있습니다:

| 예제 | 런타임 | 모델 | 아키텍처 | 노드 유형 | 주요 기능 |
|---------|---------|--------|--------------|-----------|--------------|
| **[hello-world](https://github.com/awslabs/ai-on-eks/tree/main/blueprints/inference/nvidia-dynamo/hello-world)** | CPU | N/A | 집계 | CPU | 기본 연결 테스트 |
| **[vllm](https://github.com/awslabs/ai-on-eks/tree/main/blueprints/inference/nvidia-dynamo/vllm)** | vLLM | Qwen3-0.6B | 집계 | G5 GPU | OpenAI API, 균형 잡힌 성능 |
| **[sglang](https://github.com/awslabs/ai-on-eks/tree/main/blueprints/inference/nvidia-dynamo/sglang)** | SGLang | DeepSeek-R1-Distill-8B | 집계 | G5 GPU | RadixAttention 캐싱 |
| **[trtllm](https://github.com/awslabs/ai-on-eks/tree/main/blueprints/inference/nvidia-dynamo/trtllm)** | TensorRT-LLM | DeepSeek-R1-Distill-8B | 집계 | G5 GPU | 최대 추론 성능 |
| **[multi-replica-vllm](https://github.com/awslabs/ai-on-eks/tree/main/blueprints/inference/nvidia-dynamo/multi-replica-vllm)** | vLLM | 다중 모델 | 다중 레플리카 HA | G5 GPU | KV 라우팅, 로드 밸런싱 |

### 고급 예제 (베타)

이러한 예제는 고급 Dynamo 기능을 시연하며 실험적 워크로드에 적합합니다:

| 예제 | 런타임 | 아키텍처 | 사용 사례 | 주요 기능 |
|---------|---------|--------------|----------|--------------|
| **[vllm-disagg](https://github.com/awslabs/ai-on-eks/tree/main/blueprints/inference/nvidia-dynamo/vllm-disagg)** | vLLM | 분리 | 높은 처리량 | 별도의 prefill/decode 워커 |
| **[sglang-disagg](https://github.com/awslabs/ai-on-eks/tree/main/blueprints/inference/nvidia-dynamo/sglang-disagg)** | SGLang | 분리 | 메모리 최적화 | RadixAttention + 분리 |
| **[trtllm-disagg](https://github.com/awslabs/ai-on-eks/tree/main/blueprints/inference/nvidia-dynamo/trtllm-disagg)** | TensorRT-LLM | 분리 | 초고성능 | TRT-LLM + 분리 |
| **[kv-routing](https://github.com/awslabs/ai-on-eks/tree/main/blueprints/inference/nvidia-dynamo/kv-routing)** | 다중 런타임 | 지능형 라우팅 | 캐시 최적화 | KV 인식 요청 라우팅 |

### 예제 하이라이트

**[hello-world](https://github.com/awslabs/ai-on-eks/tree/main/blueprints/inference/nvidia-dynamo/hello-world): 완벽한 시작점**
- Dynamo 플랫폼 기능 테스트를 위한 CPU 전용 배포
- 빠른 배포 (~2분)
- GPU 또는 모델 종속성 없음
- CI/CD 검증에 이상적

**[vllm](https://github.com/awslabs/ai-on-eks/tree/main/blueprints/inference/nvidia-dynamo/vllm): 대부분의 사용 사례에 권장**
- OpenAI 호환 API (`/v1/chat/completions`, `/v1/models`)
- 빠른 테스트를 위한 작은 모델 (Qwen3-0.6B)
- 프로덕션 준비 헬스 체크
- G5 GPU 최적화

**[sglang](https://github.com/awslabs/ai-on-eks/tree/main/blueprints/inference/nvidia-dynamo/sglang): 고급 캐싱 기능**
- 반복 쿼리에서 2-10배 속도 향상을 위한 RadixAttention
- 구조화된 생성 지원 (JSON/XML)
- 고급 메모리 관리
- 캐시 중심 워크로드에 적합

**[trtllm](https://github.com/awslabs/ai-on-eks/tree/main/blueprints/inference/nvidia-dynamo/trtllm): 최대 성능**
- NVIDIA TensorRT-LLM 최적화 커널
- 최고 처리량 및 최저 지연 시간
- 사용자 정의 CUDA 커널
- 프로덕션 서빙에 최적

**[multi-replica-vllm](https://github.com/awslabs/ai-on-eks/tree/main/blueprints/inference/nvidia-dynamo/multi-replica-vllm): 고가용성 배포**
- KV 라우팅이 있는 여러 독립 워커 레플리카
- 자동 로드 밸런싱 및 장애 조치
- 지능형 캐시 인식 요청 라우팅
- 고가용성이 필요한 프로덕션 워크로드에 이상적

:::info 포괄적인 테스트
모든 9개 예제는 GPU 노드가 있는 EKS 클러스터에서 철저히 테스트되고 검증되었습니다. 각 예제에는 적절한 헬스 체크, OpenAI 호환 API 엔드포인트 및 프로덕션 준비 구성이 포함됩니다. 자세한 검증 결과는 [테스트 요약](https://github.com/awslabs/ai-on-eks/blob/main/NVIDIA_Dynamo_Testing_Summary.md)을 참조하십시오.
:::

## 테스트 및 검증

### 자동화된 테스트

내장된 테스트 스크립트를 사용하여 배포를 검증합니다:

```bash
./test.sh
```

이 스크립트는:
- 프론트엔드 서비스로의 포트 포워딩 시작
- 헬스 체크, 메트릭 및 `/v1/models` 엔드포인트 테스트
- 기능 확인을 위한 샘플 추론 요청 실행

### 수동 테스트

배포에 직접 액세스:

```bash
kubectl port-forward svc/<frontend-service> 8000:8000 -n dynamo-cloud &

curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "messages": [
        {"role": "user", "content": "Explain what a Q-Bit is in quantum computing."}
    ],
    "max_tokens": 2000,
    "temperature": 0.7,
    "stream": false
}'
```

**예상 출력:**
```json
{
  "id": "1918b11a-6d98-4891-bc84-08f99de70fd0",
  "choices": [
    {
      "index": 0,
      "message": {
        "content": "A Q-bit, or qubit, is the basic unit of quantum information...",
        "role": "assistant"
      },
      "finish_reason": "stop"
    }
  ],
  "created": 1752018267,
  "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
  "object": "chat.completion"
}
```

## 모니터링 및 관측

### Grafana 대시보드

시각화를 위해 Grafana에 액세스 (기본 포트 3000):

```bash
kubectl port-forward -n kube-prometheus-stack svc/kube-prometheus-stack-grafana 3000:80
```

### Prometheus 메트릭

메트릭 수집을 위해 Prometheus에 액세스 (포트 9090):

```bash
kubectl port-forward -n kube-prometheus-stack svc/prometheus 9090:80
```

### 자동 모니터링

배포는 자동으로 다음을 생성합니다:
- **Service**: API 호출 및 메트릭을 위한 추론 그래프 노출
- **ServiceMonitor**: 메트릭 스크래핑을 위한 Prometheus 구성
- **대시보드**: 추론 모니터링을 위한 사전 구성된 Grafana 대시보드

## 고급 구성

### 버전 관리

배포는 유연한 재정의 옵션으로 Dynamo 버전을 자동으로 관리합니다:

**기본 동작:**
- `terraform/blueprint.tfvars`에서 버전 읽기 (`dynamo_stack_version = "v0.4.1"`)
- YAML 매니페스트에서 컨테이너 이미지 태그 자동 업데이트
- 소스 파일을 수정하지 않고 임시 매니페스트 생성

**재정의 옵션:**
```bash
# 환경 변수 (최우선 순위)
export DYNAMO_VERSION=v0.4.1
./deploy.sh vllm

# 인라인 재정의
DYNAMO_VERSION=v0.4.1 ./deploy.sh sglang

# terraform/blueprint.tfvars 업데이트 (영구)
dynamo_stack_version = "v0.4.1"
```

**지원되는 버전:**
- **v0.4.1**: 현재 안정 릴리스 (기본)
- 비공개 빌드의 사용자 정의 버전

### 사용자 정의 모델 배포

사용자 정의 모델을 배포하려면 `dynamo/examples/llm/configs/`의 구성 파일을 수정합니다:

1. **아키텍처 선택**: 모델 크기 및 요구 사항에 따라 선택
2. **구성 업데이트**: 적절한 YAML 파일 편집
3. **모델 파라미터 설정**: `model` 및 `served_model_name` 필드 업데이트
4. **리소스 구성**: GPU 할당 및 메모리 설정 조정

**DeepSeek-R1 70B 모델 예:**

```yaml
Common:
  model: deepseek-ai/DeepSeek-R1-Distill-Llama-70B
  max-model-len: 32768
  tensor-parallel-size: 4

Frontend:
  served_model_name: deepseek-ai/DeepSeek-R1-Distill-Llama-70B

VllmWorker:
  ServiceArgs:
    resources:
      gpu: '4'
```


### 구성 옵션

주요 구성은 `terraform/blueprint.tfvars`에 있습니다:

```hcl
# Dynamo 배포에 필요
enable_dynamo_stack = true
enable_argocd       = true

# Dynamo 플랫폼 버전
dynamo_stack_version = "v0.4.1"

# 필수 인프라 구성 요소
enable_aws_efs_csi_driver        = true
enable_aws_efa_k8s_device_plugin = true
enable_ai_ml_observability_stack = true
```

## 문제 해결

### 일반적인 문제

1. **GPU 노드 사용 불가**: Karpenter 로그 및 인스턴스 가용성 확인
2. **Pod 실패**: 리소스 제한 및 클러스터 용량 확인
3. **모델 다운로드 실패**: HuggingFace 토큰 및 네트워크 연결 확인
4. **API 503 오류**: 모델 로딩 대기 또는 워커 상태 확인

### 디버그 명령

```bash
# 클러스터 상태 확인
kubectl get nodes
kubectl get pods -n dynamo-cloud

# 로그 보기
kubectl logs -n argocd -l app.kubernetes.io/name=argocd-server
kubectl logs -n dynamo-cloud -l app=vllm-worker

# 배포 확인
kubectl get dynamographdeployment -n dynamo-cloud
kubectl describe dynamographdeployment <name> -n dynamo-cloud
```

## 노드 선택 및 사용자 정의

### 인스턴스 유형 선택

DynamoGraphDeployment에서 `nodeSelector`를 수정하여 Dynamo 구성 요소가 배포되는 Karpenter 노드 풀을 사용자 정의할 수 있습니다:

```yaml
# 예: G5 인스턴스에 GPU 워커 배포
VllmWorker:
  extraPodSpec:
    nodeSelector:
      karpenter.sh/nodepool: g5-gpu-karpenter
  resources:
    requests:
      gpu: "1"

# 예: CPU 인스턴스에 프론트엔드 배포
Frontend:
  extraPodSpec:
    nodeSelector:
      karpenter.sh/nodepool: cpu-karpenter
```

**사용 가능한 노드 풀** (기본 인프라에 구성됨):
- `g5-gpu-karpenter`: NVIDIA A10G GPU가 있는 G5 인스턴스
- `g6-gpu-karpenter`: NVIDIA L4 GPU가 있는 G6 인스턴스 (구성된 경우)
- `cpu-karpenter`: 프론트엔드용 CPU 전용 인스턴스

### 사용자 정의 개발

고급 사용자 정의 및 개발의 경우:

1. **소스 코드**: 전체 Dynamo 소스 코드는 포괄적인 문서 및 예제와 함께 [~/dynamo](https://github.com/ai-dynamo/dynamo)에서 사용 가능
2. **블루프린트 예제**: `blueprints/inference/nvidia-dynamo/` 폴더의 각 예제에는 자세한 README 파일 포함
3. **컨테이너 소스**: 모든 소스 코드는 컨테이너 내 사용자 정의를 위해 NGC 컨테이너의 `/workspace/`에 포함됨

특정 사용자 정의 지침은 각 블루프린트 예제의 개별 README 파일을 참조하십시오.

## 다중 노드 Tensor Parallelism 제한 사항

### 다중 레플리카 vs 다중 노드 이해

**다중 레플리카 배포** (예제에서 제공하는 것)와 **진정한 다중 노드 tensor parallelism** (특수 인프라가 필요)을 구분하는 것이 중요합니다:

#### 예제에서 제공하는 것 (다중 레플리카)
- **여러 독립 워커**: 각 워커 레플리카는 완전한 모델을 독립적으로 실행 (TP=1)
- **고가용성**: 개별 워커가 실패해도 서비스 계속 운영
- **로드 밸런싱**: 처리량 증가를 위해 워커 간 요청 분산
- **KV 인식 라우팅**: 성능 극대화를 위한 캐시 중복 기반 지능형 요청 라우팅
- **Kubernetes 네이티브**: 표준 Kubernetes 배포와 원활하게 작동

#### 예제에서 제공하지 않는 것 (진정한 다중 노드 TP)
- **교차 노드 모델 샤딩**: 모델이 여러 노드에 걸쳐 분할되지 않음
- **대형 모델을 위한 메모리 확장**: 각 워커가 완전한 모델을 맞춰야 함 (교차 노드 메모리 공유 없음)
- **노드 간 Tensor Parallelism**: 교차 노드 텐서 연산 없음

### 현재 Kubernetes 제한 사항

**Kubernetes는 현재 진정한 다중 노드 tensor parallelism을 지원하지 않습니다**. 여러 기술적 제약이 있습니다:

#### 인프라 요구 사항
진정한 다중 노드 tensor parallelism에는 다음이 필요합니다:
- **MPI/Slurm 환경**: 조정된 분산 모델 로딩을 위한 `mpirun` 또는 `srun` 사용
- **동기화된 초기화**: 모든 참여 노드가 동시에 시작하고 조정 유지해야 함
- **저지연 인터커넥트**: InfiniBand, NVLink 또는 유사한 고성능 네트워킹 필요
- **공유 프로세스 그룹**: K8s에서 사용할 수 없는 프로세스 그룹 관리가 필요한 분산 훈련/추론 프레임워크

#### Kubernetes가 이를 지원하지 않는 이유 (현재)

1. **Pod 격리**: Kubernetes Pod는 격리된 단위로 설계되어 교차 Pod 텐서 연산이 어려움
2. **동적 스케줄링**: K8s 동적 Pod 배치가 다중 노드 TP에 필요한 정적, 조정된 시작과 충돌
3. **네트워크 추상화**: K8s 네트워킹 추상화가 효율적인 텐서 통신에 필요한 저수준 네트워크 프리미티브를 노출하지 않음
4. **MPI 통합 누락**: Kubernetes에 네이티브 MPI 작업 관리 없음 (MPI-Operator와 같은 프로젝트가 존재하지만 추론에 널리 채택되지 않음)

### Dynamo 백엔드의 현재 지원

공식 Dynamo 문서 및 예제에 따르면 각 백엔드가 지원하는 것:

#### SGLang 다중 노드 지원
- **상태**: 다중 노드 tensor parallelism 완전 지원
- **요구 사항**: MPI 조정이 있는 Slurm 환경
- **구성**: `--nnodes`, `--node-rank` 및 `--dist-init-addr` 파라미터 사용
- **예**: TP16 (총 16 GPU)으로 4개 노드에 걸친 DeepSeek-R1
- **Kubernetes**: 지원되지 않음 - Slurm/MPI 환경 필요

```bash
# SGLang 다중 노드 예 (Slurm만)
python3 -m dynamo.sglang.worker \
  --model-path /model/ \
  --tp 16 \
  --nnodes 2 \
  --node-rank 0 \
  --dist-init-addr ${HEAD_NODE_IP}:29500
```

#### TensorRT-LLM 다중 노드 지원
- **상태**: WideEP (Wide Expert Parallelism)로 완전 지원
- **요구 사항**: MPI 런처(`srun` 또는 `mpirun`)가 있는 Slurm 환경
- **구성**: 다중 노드 TP16/EP16 구성 사용 가능
- **예**: 4x GB200 노드에 걸친 DeepSeek-R1
- **Kubernetes**: 지원되지 않음 - MPI 조정 필요

```bash
# TRT-LLM 다중 노드 예 (Slurm만)
srun --nodes=4 --ntasks-per-node=4 \
  python3 -m dynamo.trtllm \
  --model-path /model/ \
  --engine-config wide_ep_config.yaml
```

#### vLLM 다중 노드 지원
- **상태**: 현재 진정한 다중 노드 tensor parallelism 지원되지 않음
- **현재 기능**: 단일 노드 tensor parallelism만 (같은 노드의 여러 GPU)
- **구현**: 고가용성을 위한 다중 레플리카 (각 레플리카가 전체 모델 실행)
- **향후**: 향후 vLLM 릴리스에서 추가될 수 있음

### 대형 모델을 위한 해결 방법

단일 노드에 맞지 않는 모델을 실행해야 하는 경우 다음 대안을 고려하십시오:

#### 1. 고메모리 단일 노드 인스턴스
대용량 GPU 메모리가 있는 AWS 인스턴스 사용:

```yaml
# 예: 8x H100 (각 80GB = 총 640GB)이 있는 P5.48xlarge
extraPodSpec:
  nodeSelector:
    karpenter.sh/nodepool: p5-gpu-karpenter
    node.kubernetes.io/instance-type: p5.48xlarge
resources:
  requests:
    gpu: "8"
```

#### 2. 모델 최적화 기술
- **양자화**: FP16, FP8 또는 INT8 양자화된 모델 사용
- **모델 프루닝**: 덜 중요한 파라미터 제거
- **LoRA/QLoRA**: 파라미터 효율적 미세 조정된 모델 사용

#### 3. Slurm 기반 배포
진정한 다중 노드 TP가 필요한 모델의 경우 Kubernetes 외부에 배포:

```bash
# Slurm과 함께 공식 Dynamo 예제 사용
cd ~/dynamo/docs/components/backends/trtllm/
./srun_disaggregated.sh  # 8노드 분리 배포
```

#### 4. 분리된 아키텍처
더 나은 리소스 활용을 위해 분리된 예제 사용:

- **Prefill 워커**: 입력 처리 담당 (더 작은 인스턴스 가능)
- **Decode 워커**: 토큰 생성 담당 (처리량 최적화)
- **독립적 확장**: 워크로드에 따라 각 구성 요소 확장

### 향후 개발

**Kubernetes에서의 다중 노드 Tensor Parallelism**은 다음을 통해 향후 버전에서 사용 가능해질 수 있습니다:

1. **향상된 MPI 통합**: 추론 워크로드를 위한 Kubeflow의 MPI-Operator와 같은 프로젝트
2. **네이티브 K8s 지원**: Kubernetes SIG-Scheduling이 gang scheduling 및 조정된 Pod 시작 작업 중
3. **벤더 솔루션**: 클라우드 제공자가 관리형 추론을 위한 사용자 정의 솔루션 개발 가능
4. **프레임워크 진화**: 추론 프레임워크가 Kubernetes 네이티브 분산 실행 추가

### 권장 사항

**현재 배포의 경우:**

1. **소형 ~ 중형 모델 (70B 이하)**: 다중 GPU 인스턴스로 단일 노드 배포 사용
2. **고가용성 필요**: KV 라우팅이 있는 다중 레플리카 예제 사용
3. **대형 모델 (70B 이상)**: Kubernetes 외부 Slurm 기반 배포 고려
4. **최대 성능**: 최적화된 워커 비율로 분리된 아키텍처 사용

**향후 개발 모니터링:**

- Kubernetes 다중 노드 TP 업데이트는 [Dynamo 릴리스](https://github.com/ai-dynamo/dynamo/releases) 팔로우
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) 및 [vLLM](https://github.com/vllm-project/vllm) 로드맵 확인
- gang scheduling 개선을 위한 [Kubernetes SIG-Scheduling](https://github.com/kubernetes/community/tree/master/sig-scheduling) 모니터링

## 대체 배포 옵션

### 기존 EKS 클러스터의 경우

GPU 노드가 있는 EKS 클러스터가 이미 있고 더 간단한 접근 방식을 선호하는 경우:

1. **직접 Helm 설치**: [dynamo 소스 저장소](https://github.com/ai-dynamo/dynamo)에서 직접 공식 NVIDIA Dynamo Helm 차트 사용
2. **수동 설정**: Kubernetes 배포에 대한 업스트림 NVIDIA Dynamo 문서 팔로우
3. **사용자 정의 통합**: 기존 인프라에 Dynamo 구성 요소 통합

### 이 블루프린트를 사용하는 이유는?

이 블루프린트는 다음을 원하는 사용자를 위해 설계되었습니다:
- **완전한 인프라**: VPC에서 실행 중인 추론까지 엔드투엔드 설정
- **프로덕션 준비**: 엔터프라이즈급 모니터링, 보안 및 확장성
- **AWS 통합**: EKS, ECR, EFA 및 기타 AWS 서비스에 최적화
- **모범 사례**: ai-on-eks 패턴 및 AWS 권장 사항 준수

## 참조

### 공식 NVIDIA 리소스

**문서:**
- [NVIDIA Dynamo 공식 문서](https://docs.nvidia.com/dynamo/latest/): 전체 플랫폼 문서
- [NVIDIA Developer Blog](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/): 소개 및 아키텍처 개요
- [NVIDIA Dynamo 제품 페이지](https://developer.nvidia.com/dynamo): 공식 제품 정보

**소스 코드:**
- [NVIDIA Dynamo GitHub](https://github.com/ai-dynamo/dynamo): 소스 코드가 있는 메인 저장소
- [NVIDIA NIXL Library](https://github.com/ai-dynamo/nixl): 저지연 통신을 위한 NVIDIA Inference Xfer Library

**컨테이너 이미지 및 Helm 차트:**
- [Dynamo Collection (NGC)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo): Dynamo 리소스의 전체 컬렉션
- [Dynamo Platform Helm Chart](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/helm-charts/dynamo-platform): 공식 Kubernetes 배포
- [vLLM Runtime Container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/vllm-runtime): vLLM 백엔드 (v0.4.1)
- [SGLang Runtime Container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/sglang-runtime): SGLang 백엔드 (v0.4.1)
- [TensorRT-LLM Runtime Container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/trtllm-runtime): TRT-LLM 백엔드 (v0.4.1)

### AI-on-EKS 블루프린트 리소스

**인프라 및 예제:**
- [AI-on-EKS Repository](https://github.com/awslabs/ai-on-eks): 메인 블루프린트 저장소
- [Dynamo Blueprint](https://github.com/awslabs/ai-on-eks/tree/main/blueprints/inference/nvidia-dynamo): 예제가 포함된 전체 블루프린트
- [Infrastructure Code](https://github.com/awslabs/ai-on-eks/tree/main/infra/nvidia-dynamo): Terraform 및 배포 스크립트

**예제 문서:**
- [Hello World](https://github.com/awslabs/ai-on-eks/tree/main/blueprints/inference/nvidia-dynamo/hello-world/README.md): CPU 전용 테스트 예제
- [vLLM Example](https://github.com/awslabs/ai-on-eks/tree/main/blueprints/inference/nvidia-dynamo/vllm/README.md): vLLM 집계 서빙
- [SGLang Example](https://github.com/awslabs/ai-on-eks/tree/main/blueprints/inference/nvidia-dynamo/sglang/README.md): RadixAttention 캐싱
- [TensorRT-LLM Example](https://github.com/awslabs/ai-on-eks/tree/main/blueprints/inference/nvidia-dynamo/trtllm/README.md): 최적화 추론
- [Multi-Replica vLLM](https://github.com/awslabs/ai-on-eks/tree/main/blueprints/inference/nvidia-dynamo/multi-replica-vllm/README.md): 고가용성 배포

### 관련 기술

**추론 프레임워크:**
- [vLLM](https://github.com/vllm-project/vllm): 고처리량 LLM 추론 엔진
- [SGLang](https://github.com/sgl-project/sglang): RadixAttention이 있는 구조화된 생성
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM): NVIDIA의 최적화된 추론 라이브러리

**Kubernetes 및 AWS:**
- [Amazon EKS](https://aws.amazon.com/eks/): 관리형 Kubernetes 서비스
- [Karpenter](https://karpenter.sh/): Kubernetes 노드 오토스케일링
- [ArgoCD](https://argo-cd.readthedocs.io/): GitOps 지속적 배포

## 다음 단계

1. **예제 탐색**: GitHub 저장소의 예제 폴더 확인
2. **배포 확장**: 대형 모델을 위한 다중 노드 설정 구성
3. **애플리케이션 통합**: 애플리케이션을 추론 엔드포인트에 연결
4. **성능 모니터링**: 지속적인 모니터링을 위해 Grafana 대시보드 사용
5. **비용 최적화**: 오토스케일링 및 리소스 최적화 구현

## 정리

NVIDIA Dynamo 배포가 완료되면 통합 정리 스크립트를 사용하여 모든 리소스를 제거합니다:

```bash
cd infra/nvidia-dynamo
./cleanup.sh
```

**정리되는 것 (올바른 순서로):**
- **Dynamo 예제**: 배포된 모든 추론 그래프 및 워크로드
- **Dynamo Platform**: Operator, API Store 및 지원 서비스
- **ArgoCD 애플리케이션**: GitOps 관리 리소스
- **Kubernetes 리소스**: 네임스페이스, 시크릿 및 구성
- **인프라**: EKS 클러스터, VPC, 보안 그룹 및 모든 AWS 리소스
- **비용 최적화**: 잔류 리소스가 계속 청구되지 않도록 보장

**기능:**
- **지능형 순서**: 종속성을 올바른 순서로 정리
- **안전 검사**: 삭제 시도 전 리소스 존재 확인
- **진행 상황 피드백**: 정리 진행 상황 및 발생한 문제 표시
- **완전 제거**: 수동 정리 단계 불필요

**소요 시간**: 전체 인프라 해제에 ~10-15분

이 배포는 Karpenter 자동 확장, EFA 네트워킹 및 원활한 AWS 서비스 통합을 포함한 엔터프라이즈급 기능과 함께 Amazon EKS에서 프로덕션 준비된 NVIDIA Dynamo 환경을 제공합니다.
