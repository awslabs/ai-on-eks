---
sidebar_label: EKS에서 Inference Perf 실행하기
---

# EKS에서 Inference Perf 실행하기

Inference Perf는 Kubernetes에서 LLM 추론 엔드포인트 성능을 측정하기 위해 특별히 설계된 GenAI 추론 성능 벤치마킹 도구입니다. Kubernetes Job으로 실행되며 표준화된 메트릭으로 여러 모델 서버([vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [TGI](https://github.com/huggingface/text-generation-inference))를 지원합니다.

왜 [Job](https://kubernetes.io/docs/concepts/workloads/controllers/job/)을 사용하나요? Job은 한 번 실행되고 완료되면 종료되므로 벤치마킹 작업에 이상적입니다. 결과는 로컬 또는 클라우드 스토리지([S3](https://aws.amazon.com/s3/))에 저장됩니다.


## 사전 요구 사항

* kubectl 접근 가능한 Kubernetes 클러스터 (버전 1.21+)
* OpenAI 호환 API(vLLM, SGLang, TGI 또는 호환 가능)가 있는 배포된 추론 엔드포인트
* 벤치마크 실행을 위한 네임스페이스
* 컨테이너 이미지: quay.io/inference-perf/inference-perf:v0.2.0
* (선택 사항) 토크나이저 다운로드를 위한 HuggingFace 토큰
* (선택 사항) S3 스토리지를 위한 AWS 자격 증명


## 모델별 종속성

주의: 다른 모델은 다른 토크나이저 패키지를 필요로 합니다:

| 모델 패밀리 | sentencepiece 필요? | 예시 |
|--------------|------------------------|----------|
| Mistral (모든 버전) | 예 | mistralai/Mistral-7B-Instruct-v0.3 |
| Llama 2 | 예 | meta-llama/Llama-2-7b-hf |
| Llama 3.1 | 예 | meta-llama/Meta-Llama-3.1-8B |
| SmolLM2 | 아니오 | HuggingFaceTB/SmolLM2-135M-Instruct |
| GPT 모델 | 아니오 | 다양한 GPT 변형 |

Mistral 또는 Llama 모델을 사용하는 경우 sentencepiece 패키지를 설치해야 합니다. 구현에 대해서는 아래의 "모델 종속성 처리" 섹션을 참조하십시오.


## 추론 벤치마크 프레임워크 아키텍처 이해

배포하기 전에 벤치마크 테스트를 정의하는 주요 구성 구성 요소를 이해하는 것이 중요합니다.

### API 구성

도구가 추론 엔드포인트와 통신하는 방법을 정의합니다. completion 또는 chat API를 사용하는지, 스트리밍이 활성화되어 있는지(TTFT 및 ITL 메트릭 측정에 필요) 지정합니다.

```yaml
api:
  type: completion # completion 또는 chat
  streaming: true # TTFT/ITL 메트릭을 위해 활성화
```

**올바른 API 유형 결정:**

inference-charts 배포는 모델의 기능에 따라 API 엔드포인트를 자동으로 구성합니다. 사용 가능한 엔드포인트를 식별하려면:

**방법 1: vLLM 배포 로그 확인 (권장)**

vLLM 서버 로그를 확인하여 시작 시 활성화된 API 엔드포인트를 확인합니다:

```bash
# 활성화된 엔드포인트를 보여주는 vLLM 시작 로그 보기
kubectl logs -n default -l app.kubernetes.io/name=inference-charts --tail=100 | grep -i "route\|endpoint\|application"
```

활성화된 라우트를 나타내는 출력을 찾습니다:
- `Route: /v1/completions` → `type: completion` 사용
- `Route: /v1/chat/completions` → `type: chat` 사용

두 라우트가 모두 나타나면 `completion`을 사용합니다 (벤치마킹에 더 간단).

**방법 2: 모델 기능 확인 (선택 사항)**

모델의 이론적 기능을 이해하려면 모델의 Hugging Face 모델 카드를 검토하십시오. 정의된 채팅 템플릿이 있는 모델은 일반적으로 chat completion API를 지원하지만 실제 배포 구성이 활성화된 것을 결정합니다.

**참고:** OpenAI completion API (`v1/completions`)는 OpenAI에서 더 이상 사용되지 않지만 vLLM, SGLang 및 TGI에서 널리 지원됩니다. 대부분의 inference-charts 배포는 추가 구성 없이 기본적으로 활성화합니다.

### 데이터 생성

추론 엔드포인트로 전송되는 데이터를 제어합니다. 실제 데이터셋(ShareGPT) 또는 제어된 분포를 가진 합성 데이터를 사용할 수 있습니다. 합성 데이터는 테스트를 위해 특정 입력/출력 길이 패턴이 필요할 때 유용합니다.

```yaml
data:

  type: synthetic # shareGPT, synthetic, random, shared_prefix 등

  input_distribution:
    mean: 512      # 토큰 단위 평균 입력 프롬프트 길이
    std_dev: 128   # 프롬프트 길이의 변동 (평균 ±128 토큰 내 68%)
    min: 128       # 최소 입력 토큰 (분포 하한 클리핑)
    max: 2048      # 최대 입력 토큰 (분포 상한 클리핑)

  output_distribution:
    mean: 256      # 토큰 단위 평균 생성 응답 길이
    std_dev: 64    # 응답 길이의 변동 (평균 ±64 토큰 내 68%)
    min: 32        # 최소 출력 토큰 (분포 하한 클리핑)
    max: 512       # 최대 출력 토큰 (분포 상한 클리핑)
```


### 부하 생성

부하 패턴을 정의합니다 - 초당 요청 수와 기간. 여러 단계를 사용하여 다른 부하 수준을 테스트하거나 자동 포화 감지를 위해 sweep 모드를 사용할 수 있습니다.

```yaml
load:
  type: constant              # 균일한 도착(예측 가능한 부하)에는 'constant', 버스트 트래픽(현실적인 프로덕션)에는 'poisson' 사용
  stages:
    - rate: 10                # 초당 요청 수 (QPS) - 더 높은 처리량을 테스트하려면 증가, 기준선/최소 부하에는 감소
      duration: 300           # 이 비율을 유지할 시간(초) - 더 긴 기간(300-600초)은 안정적인 측정을 보장
  num_workers: 4              # 부하를 생성하는 동시 워커 - inference-perf가 목표 비율을 달성할 수 없는 경우 증가 (결과에서 스케줄링 지연 확인)
```

**num_workers 참고:** 이것은 동시 사용자가 아닌 벤치마크 도구의 내부 병렬성을 제어합니다. 기본값 4는 대부분의 시나리오에서 작동합니다. 결과에서 높은 `schedule_delay` (> 10ms)가 표시되어 도구가 목표 비율을 유지할 수 없음을 나타내는 경우에만 증가시키십시오.

### 서버 구성

추론 엔드포인트 세부 정보를 지정합니다 - 서버 유형, 모델 이름 및 URL.

```yaml
server:

  type: vllm # vllm, sglang 또는 tgi

  model_name: qwen3-8b

  base_url: http://qwen3-vllm.default:8000

  ignore_eos: true
```

### 스토리지 구성

벤치마크 결과가 저장되는 위치를 결정합니다. 로컬 스토리지는 Pod 파일 시스템에 저장(수동 복사 필요)하고, S3 스토리지는 결과를 AWS 버킷에 자동으로 유지합니다.

```yaml
storage:

  local_storage: # 기본값: Pod에 저장

    path: "reports-results"

  # 주의: local_storage 결과는 Pod 종료 시 손실됩니다
  # 결과를 검색하려면 Job args에 '&& sleep infinity'를 추가하고 다음을 사용합니다:
  # kubectl cp <pod-name>:/workspace/reports-* ./local-results -n benchmarking

  # 또는

  simple_storage_service: # S3: 자동 지속성

    bucket_name: "my-results-bucket"

    path: "inference-perf/results"
```

### 메트릭 수집 (선택 사항)

추론 서버가 메트릭을 노출하는 경우 Prometheus에서 고급 메트릭 수집을 활성화합니다.

```yaml
metrics:

  type: prometheus

  prometheus:

    url: http://kube-prometheus-stack-prometheus.monitoring:9090 # ai-on-eks Path A용; 사용자 정의 Prometheus의 경우 서비스 이름/네임스페이스 조정

    scrape_interval: 15
```

**참고:** Prometheus URL은 Kubernetes DNS 형식을 사용합니다: `http://<service-name>.<namespace>:<port>`. Prometheus가 다른 네임스페이스(예: `monitoring`, `observability`)에 배포된 경우 URL을 적절히 업데이트하십시오. 벤치마크 Job은 `benchmarking` 네임스페이스에서 실행되므로 크로스 네임스페이스 서비스 접근을 지정해야 합니다.

## 재현 가능한 결과를 위한 인프라 토폴로지

여러 실행에 걸쳐 정확하고 비교 가능한 벤치마크를 위해 inference-perf Job은 추론 배포와 **반드시** 동일한 AZ에 배치되어야 합니다.

### 중요한 이유:

적절한 배치 없이는 벤치마크 결과가 신뢰할 수 없게 됩니다:

* 크로스 AZ 네트워크 지연은 요청당 1-2ms를 추가합니다
* 벤치마크 실행에 따라 결과가 예측 불가능하게 변합니다
* 성능 변화가 실제인지 인프라 배치 때문인지 결정할 수 없습니다
* 최적화 결정이 불가능해집니다

### 문제의 예:

```
첫 번째 벤치마크 실행:
- us-west-2a의 벤치마크 Pod → us-west-2a의 추론 Pod
- 결과: TTFT = 800ms

두 번째 벤치마크 실행 (Pod 재시작 후):
- us-west-2b의 벤치마크 Pod → us-west-2a의 추론 Pod
- 결과: TTFT = 850ms
```

50ms 차이는 크로스 AZ 지연이지 실제 성능 변화가 아닙니다.

**크로스 AZ 테스트 참고:** 동일 AZ 배치가 기준선 벤치마킹 및 성능 최적화에 권장되지만, 크로스 AZ 테스트는 추론 서비스가 여러 가용 영역에 걸쳐 있는 고가용성(HA) 배포를 검증하는 데 유용합니다. 프로덕션 배포가 내결함성을 위해 다중 AZ 로드 밸런싱을 사용하는 경우, 크로스 AZ 배치로 별도의 벤치마크를 수행하여 영역별 라우팅 중 사용자가 경험할 수 있는 지연 영향을 이해하십시오.

### 필수 구성:

이 가이드의 모든 벤치마크 Job 예제에는 표준 Kubernetes 토폴로지 레이블 `topology.kubernetes.io/zone`을 사용하여 동일 AZ 배치를 적용하는 `affinity` 구성이 포함되어 있습니다:

```yaml
spec:
  template:
    spec:
      affinity:
        podAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchLabels:
                app.kubernetes.io/component: qwen3-vllm
            topologyKey: topology.kubernetes.io/zone
```


**중요:** `matchLabels`는 실제 vLLM 배포 레이블과 일치해야 합니다. 다음으로 배포의 Pod 레이블을 확인하십시오:

```bash
kubectl get deployment qwen3-vllm -n default -o jsonpath='{.spec.template.metadata.labels}' && echo
```


일반적인 레이블 패턴:

* 표준 배포: `app: <service-name>` (간단한 패턴)
* inference-charts 배포: `app.kubernetes.io/component: <service-name>` (이 가이드의 예제에서 사용)
* 기타 Helm 차트: `app.kubernetes.io/name: <service-name>`

예제의 `matchLabels` 섹션을 배포의 실제 Pod 레이블과 일치하도록 업데이트하십시오.

### 확인:

배포 후 두 Pod가 동일한 AZ에 있는지 확인합니다:

```bash
# 두 Pod 모두 확인 - 동일한 영역을 표시해야 합니다
kubectl get pods -n default -o wide -l app.kubernetes.io/component=qwen3-vllm
kubectl get pods -n benchmarking -o wide -l app=inference-perf

# 예상 출력 - 둘 다 동일한 영역:
# qwen3-vllm-xxx      ip-10-0-1-100.us-west-2a...
# inference-perf-yyy  ip-10-0-1-200.us-west-2a...
```


### 선택 사항: 인스턴스 유형 일관성

**벤치마크 Pod의 인스턴스 크기 조정**

벤치마크 Pod는 GPU 기반 추론 배포와 별도의 CPU 노드에서 실행됩니다. m6i.2xlarge 인스턴스 유형(8 vCPU, 32 GB RAM)은 GPU 노드 리소스와 경쟁하지 않고 부하 생성에 충분한 용량을 제공합니다.

**중요:** Pod 어피니티 구성(`topology.kubernetes.io/zone`)은 두 Pod가 동일한 물리적 노드가 아닌 동일한 가용 영역에 있도록 보장합니다. 클러스터에는 다음 두 가지 모두를 위한 용량이 있어야 합니다:
- 추론을 위한 GPU 노드 (예: Qwen3-8B와 같은 모델용 g5.2xlarge)
- 벤치마킹을 위한 CPU 노드 (예: m6i.2xlarge)

Karpenter를 사용하는 경우 동일한 AZ에서 적절한 노드 유형을 자동으로 프로비저닝합니다.

최대 재현성(기준선 벤치마크, CI/CD 파이프라인)을 위해 인스턴스 유형을 지정할 수 있습니다:

```yaml
spec:
  template:
    spec:
      nodeSelector:
        node.kubernetes.io/instance-type: m6i.2xlarge
      affinity:
        podAffinity:
          # ... 위와 동일
```


**인스턴스 유형 셀렉터를 사용해야 하는 경우:**

* 문서화를 위한 벤치마크 기준선 생성
* 일관된 결과가 필요한 CI/CD 파이프라인
* Karpenter가 다른 인스턴스 패밀리를 프로비저닝하는 것을 방지

**필요하지 않은 경우:**

* 동종 CPU 노드 풀
* 비교 테스트 (동일한 인프라에서 전/후)

### 문제 해결:

벤치마크 Job이 `Pending` 상태로 유지되는 경우:

```bash
kubectl describe pod -n benchmarking <pod-name>
```


일반적인 문제:

* **대상 AZ에 용량 없음**: 클러스터를 확장하거나 `preferredDuringSchedulingIgnoredDuringExecution` 사용
* **레이블 불일치**: 배포 레이블이 podAffinity 셀렉터와 일치하는지 확인
