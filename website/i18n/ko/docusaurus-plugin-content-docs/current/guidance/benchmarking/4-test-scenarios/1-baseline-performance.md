---
sidebar_label: 시나리오 1 - 베이스라인 성능
---

# 시나리오 1: 베이스라인 성능

## 이 시나리오를 사용해야 하는 경우:
경쟁 없이 시스템의 최적 성능을 확립할 때 베이스라인 테스트를 사용하십시오. 본질적으로 인프라의 바이탈 사인을 측정하는 것입니다. 이것은 용량 계획이나 최적화 작업 전 시작점으로, 새 엔드포인트를 방금 배포했거나 인프라를 변경했을 때 이상적입니다. "이 시스템이 제공할 수 있는 최고의 성능은 무엇인가?"라는 질문에 대기열이나 리소스 경쟁 없이 답하며, 향후 모든 테스트를 위한 깨끗한 참조 지점을 제공합니다.

## 배포

### Helm 차트 사용 (권장)

```bash
# AI on EKS Helm 저장소 추가
helm repo add ai-on-eks https://awslabs.github.io/ai-on-eks-charts/
helm repo update

# 베이스라인 시나리오 설치
helm install baseline-test ai-on-eks/benchmark-charts \
  --set benchmark.scenario=baseline \
  --set benchmark.target.baseUrl=http://qwen3-vllm.default:8000 \
  --set benchmark.target.modelName=qwen3-8b \
  --set benchmark.target.tokenizerPath=Qwen/Qwen3-8B \
  --namespace benchmarking --create-namespace

# 진행 상황 모니터링
kubectl logs -n benchmarking -l benchmark.scenario=baseline -f
```

### 구성 사용자 정의

`--set` 또는 사용자 정의 values 파일을 사용하여 특정 값을 재정의합니다:

```bash
# 테스트 기간 또는 리소스 조정
helm install baseline-test ai-on-eks/benchmark-charts \
  --set benchmark.scenario=baseline \
  --set benchmark.target.baseUrl=http://your-model.your-namespace:8000 \
  --set benchmark.scenarios.baseline.load.stages[0].duration=600 \
  --set benchmark.resources.main.requests.cpu=4 \
  --namespace benchmarking
```

또는 사용자 정의 `my-values.yaml` 생성:

```yaml
benchmark:
  scenario: baseline
  target:
    baseUrl: http://your-model.your-namespace:8000
    modelName: your-model-name
  scenarios:
    baseline:
      load:
        stages:
          - rate: 1
            duration: 600  # 더 긴 테스트
```

```bash
helm install baseline-test ai-on-eks/benchmark-charts -f my-values.yaml -n benchmarking
```

## 주요 구성:

* 고정 길이 합성 데이터 (512 입력 / 128 출력 토큰)
* 300초 동안 1 QPS의 일정한 부하
* 스트리밍 활성화
* Pod 어피니티가 추론 Pod와 동일 AZ 배치를 보장

## 결과 이해:
1 QPS에서의 TTFT와 ITL은 이론적 최소 지연 시간을 나타내며, 대기열이나 경쟁 없이 시스템이 응답할 수 있는 절대적으로 가장 빠른 속도입니다. 베이스라인 TTFT가 800ms이면 복제본 추가, 로드 밸런서 또는 오토스케일링과 같은 최적화와 관계없이 사용자는 더 빠른 응답 시간을 볼 수 없습니다. 이러한 것들은 단일 요청 속도가 아닌 **처리량과 동시성**을 개선하기 때문입니다. 이러한 메트릭에 성능 하한으로 집중하십시오: 스케줄 지연은 거의 0(`<10ms`)이어야 하며, 편차가 있으면 테스트 러너 자체에 더 많은 리소스가 필요함을 나타냅니다. 베이스라인 수치를 서비스 수준 계약(SLA) 목표와 비교하십시오; 베이스라인 성능이 요구 사항을 충족하지 않으면 규모에 대해 걱정하기 전에 모델/하드웨어 최적화가 필요합니다. 용량을 추가해도 근본적인 추론 속도는 개선되지 않기 때문입니다.

<details>
<summary><strong>대안: 원시 Kubernetes YAML</strong> (교육 목적 또는 사용자 정의 배포용)</summary>

Helm을 사용하지 않거나 값을 넘어서 사용자 정의해야 하는 경우 완전한 Kubernetes 매니페스트는 다음과 같습니다:

```yaml
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: inference-perf-baseline
  namespace: benchmarking
data:
  config.yml: |
    api:
      type: completion
      streaming: true

    data:
      type: synthetic
      input_distribution:
        mean: 512
        std_dev: 0
        min: 512
        max: 512
      output_distribution:
        mean: 128
        std_dev: 0
        min: 128
        max: 128

    load:
      type: constant
      stages:
        - rate: 1
          duration: 300
      num_workers: 4

    server:
      type: vllm
      model_name: qwen3-8b
      base_url: http://qwen3-vllm.default:8000
      ignore_eos: true

    tokenizer:
      pretrained_model_name_or_path: Qwen/Qwen3-8B

    storage:
      simple_storage_service:
        bucket_name: "inference-perf-results"
        path: "baseline-test/results"
---
apiVersion: batch/v1
kind: Job
metadata:
  name: inference-perf-baseline
  namespace: benchmarking
spec:
  backoffLimit: 2
  ttlSecondsAfterFinished: 3600
  template:
    spec:
      restartPolicy: Never
      serviceAccountName: inference-perf-sa

      affinity:
        podAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchLabels:
                app.kubernetes.io/component: qwen3-vllm
            topologyKey: topology.kubernetes.io/zone

      containers:
      - name: inference-perf
        image: quay.io/inference-perf/inference-perf:v0.2.0
        command: ["/bin/sh", "-c"]
        args:
        - |
          inference-perf --config_file /workspace/config.yml
        volumeMounts:
        - name: config
          mountPath: /workspace/config.yml
          subPath: config.yml
        resources:
          requests:
            cpu: "2"
            memory: "4Gi"
          limits:
            cpu: "4"
            memory: "8Gi"

      volumes:
      - name: config
        configMap:
          name: inference-perf-baseline
```

다음으로 적용: `kubectl apply -f 01-scenario-baseline.yaml`

</details>
