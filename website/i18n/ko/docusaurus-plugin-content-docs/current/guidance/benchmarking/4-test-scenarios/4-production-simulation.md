---
sidebar_label: 시나리오 4 - 프로덕션 시뮬레이션
---

# 시나리오 4: 프로덕션 시뮬레이션

## 이 시나리오를 사용해야 하는 경우:
출시 전 최종 검증으로 프로덕션 시뮬레이션을 배포하십시오; 균일한 부하 대신 가변 요청 크기와 Poisson(버스트) 도착으로 실제 트래픽 혼란을 복제합니다. 베이스라인 및 포화 테스트를 기반으로 최적화한 후 "현실적인 조건에서 사용자가 좋은 경험을 할 것인가?"라는 질문에 답하기 위해 이것을 사용하십시오. 실제 프로덕션 트래픽은 시계처럼 도착하는 동일한 512 토큰 요청으로 구성되지 않습니다; 사용자는 무작위 간격으로 다양한 길이를 보내며, 이 테스트는 시스템이 SLA 설정을 위한 허용 가능한 백분위수 지연 시간을 유지하면서 그 이질성을 처리하는지 검증합니다.

## 배포

### Helm 차트 사용 (권장)

```bash
# AI on EKS Helm 저장소 추가
helm repo add ai-on-eks https://awslabs.github.io/ai-on-eks-charts/
helm repo update

# 프로덕션 시나리오 설치
helm install production-sim ai-on-eks/benchmark-charts \
  --set benchmark.scenario=production \
  --set benchmark.target.baseUrl=http://qwen3-vllm.default:8000 \
  --set benchmark.target.modelName=qwen3-8b \
  --set benchmark.target.tokenizerPath=Qwen/Qwen3-8B \
  --namespace benchmarking --create-namespace

# 진행 상황 모니터링 - 버스트 트래픽으로 인한 가변 지연 시간 예상
kubectl logs -n benchmarking -l benchmark.scenario=production -f
```

### 트래픽 패턴 사용자 정의

버스트 비율 및 변동성 조정:

```yaml
# custom-production.yaml
benchmark:
  scenario: production
  target:
    baseUrl: http://your-model.your-namespace:8000
  scenarios:
    production:
      data:
        input:
          mean: 2048          # 더 긴 평균 프롬프트
          stdDev: 1024        # 더 높은 변동성
          min: 256
          max: 8192
      load:
        type: poisson         # 버스트 도착 유지
        stages:
          - rate: 20          # 더 높은 목표 QPS
            duration: 900     # 더 긴 테스트 (15분)
```

```bash
helm install production-sim ai-on-eks/benchmark-charts -f custom-production.yaml -n benchmarking
```

## 주요 구성:

* 가변 합성 데이터 (입력/출력에 대한 가우시안 분포)
* 넓은 토큰 분포 (평균 1024/512, 높은 분산)
* 균일한 부하 대신 Poisson(버스트) 도착
* 스트리밍 활성화
* 8개의 동시 워커

## 결과 이해:
P99 및 P95 지연 시간에만 집중하십시오; 이러한 백분위수는 99%와 95%의 사용자가 경험하는 최악의 경험을 나타내며, 나쁜 테일 성능을 숨기는 평균과 다릅니다. 넓은 입력/출력 분포는 자연스러운 변동성을 생성하므로 베이스라인 테스트보다 더 높은 분산을 예상하십시오; 이것은 정상이며 프로덕션 현실을 반영합니다. Poisson 버스트는 지속 가능한 평균 비율에서도 임시 대기열 축적을 유발하므로, P99가 균일 부하 테스트가 제안한 것보다 상당히 나쁘면 예상보다 더 많은 여유가 필요합니다. 평균이 아닌 이러한 현실적인 백분위수를 기반으로 SLA를 설정하십시오; P99 TTFT가 1200ms이면 평균이 400ms일 수 있어도 1초 미만 지연 시간을 약속하지 마십시오.

<details>
<summary><strong>대안: 원시 Kubernetes YAML</strong></summary>

```yaml
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: inference-perf-production
  namespace: benchmarking
data:
  config.yml: |
    api:
      type: completion
      streaming: true

    data:
      type: synthetic
      input_distribution:
        mean: 1024
        std_dev: 512
        min: 128
        max: 4096
      output_distribution:
        mean: 512
        std_dev: 256
        min: 50
        max: 2048

    load:
      type: poisson  # 현실적인 버스트 도착
      stages:
        - rate: 15
          duration: 600
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
        path: "production-sim/results"
---
apiVersion: batch/v1
kind: Job
metadata:
  name: inference-perf-production
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
          name: inference-perf-production
```

</details>
