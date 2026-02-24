---
sidebar_label: 시나리오 2 - 포화 테스트
---

# 시나리오 2: 포화 테스트

## 이 시나리오를 사용해야 하는 경우:
성능이 저하되기 전 시스템의 최대 지속 가능한 처리량을 경험적으로 결정해야 할 때 다단계 포화 테스트를 사용하십시오. 이것은 프로덕션 출시 전, 용량을 계획할 때 또는 오토스케일링 임계값을 설정할 때 중요합니다. "신뢰성 있게 처리할 수 있는 최고 QPS는 무엇인가?"라는 질문에 답합니다. 체계적인 부하 증가를 통해 지연 시간이 상승하거나 오류가 나타나는 지점을 관찰하여 마케팅 자료와 이론적 계산이 종종 과대평가하는 진정한 용량 한계를 밝힙니다.

## 배포

### Helm 차트 사용 (권장)

```bash
# AI on EKS Helm 저장소 추가
helm repo add ai-on-eks https://awslabs.github.io/ai-on-eks-charts/
helm repo update

# 포화 시나리오 설치
helm install saturation-test ai-on-eks/benchmark-charts \
  --set benchmark.scenario=saturation \
  --set benchmark.target.baseUrl=http://qwen3-vllm.default:8000 \
  --set benchmark.target.modelName=qwen3-8b \
  --set benchmark.target.tokenizerPath=Qwen/Qwen3-8B \
  --namespace benchmarking --create-namespace

# 여러 단계를 통한 진행 상황 모니터링
kubectl logs -n benchmarking -l benchmark.scenario=saturation -f
```

### 부하 단계 사용자 정의

예상 용량에 맞게 QPS 단계를 조정합니다:

```yaml
# custom-saturation.yaml
benchmark:
  scenario: saturation
  target:
    baseUrl: http://your-model.your-namespace:8000
  scenarios:
    saturation:
      load:
        stages:
          - rate: 10
            duration: 180
          - rate: 25
            duration: 180
          - rate: 50
            duration: 180
          - rate: 75
            duration: 180
```

```bash
helm install saturation-test ai-on-eks/benchmark-charts -f custom-saturation.yaml -n benchmarking
```

## 주요 구성:

* 가변 합성 데이터 분포 (평균 512/256 토큰, 현실적인 분산)
* 다단계 일정 부하: 5 → 10 → 20 → 40 QPS (각 3분)
* 스트리밍 활성화
* 8개의 동시 워커

## 결과 이해:
모든 단계에 걸쳐 P50, P95 및 P99 지연 시간을 플로팅하여 포화 지점을 시각적으로 식별하십시오. 백분위수가 급격히 갈라지거나 오류율이 급증하는 단계를 찾으십시오. 지연 시간 곡선의 "무릎"(하키 스틱처럼 위로 꺾이는 곳)은 시스템이 처리한 이론적 최대 QPS가 아닌 실질적인 용량 한계를 나타냅니다. 트래픽 급증에 대한 여유를 유지하기 위해 이 포화 지점보다 20-30% 낮게 프로덕션 목표를 설정하십시오; 포화가 35 QPS에서 발생하면 24-28 QPS 지속 부하를 목표로 하십시오. 동일한 테스트 단계를 사용하여 다른 모델 구성이나 하드웨어 설정을 비교하여 벤더 주장이 아닌 데이터에 기반한 객관적인 확장 결정을 내리십시오.

<details>
<summary><strong>대안: 원시 Kubernetes YAML</strong></summary>

```yaml
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: inference-perf-saturation
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
        std_dev: 128
        min: 128
        max: 2048
      output_distribution:
        mean: 256
        std_dev: 64
        min: 32
        max: 512

    load:
      type: constant
      stages:
        - rate: 5
          duration: 180
        - rate: 10
          duration: 180
        - rate: 20
          duration: 180
        - rate: 40
          duration: 180
      num_workers: 8

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
        path: "saturation-test/results"
---
apiVersion: batch/v1
kind: Job
metadata:
  name: inference-perf-saturation
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
          name: inference-perf-saturation
```

</details>
