---
sidebar_label: 시나리오 3 - 자동 포화 감지
---

# 시나리오 3: 자동 포화 감지

## 이 시나리오를 사용해야 하는 경우:
적절한 QPS 테스트 단계를 수동으로 추측하고 싶지 않을 때 자동화된 용량 검색을 위해 sweep 모드를 사용하십시오. 초기 배포, CI/CD 파이프라인 또는 인프라 변경 후 빠른 용량 재검증에 이상적입니다. 도구가 시스템을 플러딩하여 포화를 경험적으로 결정한 다음 해당 중요 지점 주변에 클러스터된 지능형 테스트 단계를 자동으로 생성합니다. 이것은 테스트 설계에서 인간 편향을 제거하고 다양한 환경과 팀에 걸쳐 일관되고 재현 가능한 방법론을 보장하지만, 과학적 자동화를 위해 세밀한 제어를 교환합니다.

특정 부하 목표를 검증해야 하거나(예: "20 QPS를 처리할 수 있는가?") 프로덕션 환경을 위한 예측 가능한 테스트 단계를 원할 때 **시나리오 2를 선택**하십시오.

알 수 없는 용량 한계를 발견하거나 특정 QPS 값을 테스트하는 것보다 일관된 자동화 방법론이 더 중요할 때 **시나리오 3을 선택**하십시오.

## 배포

### Helm 차트 사용 (권장)

```bash
# AI on EKS Helm 저장소 추가
helm repo add ai-on-eks https://awslabs.github.io/ai-on-eks-charts/
helm repo update

# sweep 시나리오 설치
helm install sweep-test ai-on-eks/benchmark-charts \
  --set benchmark.scenario=sweep \
  --set benchmark.target.baseUrl=http://qwen3-vllm.default:8000 \
  --set benchmark.target.modelName=qwen3-8b \
  --set benchmark.target.tokenizerPath=Qwen/Qwen3-8B \
  --namespace benchmarking --create-namespace

# 진행 상황 모니터링 - 로그에서 자동 단계 생성 확인
kubectl logs -n benchmarking -l benchmark.scenario=sweep -f
```

### Sweep 매개변수 사용자 정의

포화 프로브 설정 조정:

```yaml
# custom-sweep.yaml
benchmark:
  scenario: sweep
  target:
    baseUrl: http://your-model.your-namespace:8000
  scenarios:
    sweep:
      load:
        sweep:
          numRequests: 3000        # 더 큰 시스템을 위한 더 많은 요청
          timeout: 90              # 더 긴 프로브 시간
          numStages: 7             # 더 많은 테스트 단계
          stageDuration: 240       # 더 긴 단계 기간
          saturationPercentile: 99 # 더 보수적인 추정
```

```bash
helm install sweep-test ai-on-eks/benchmark-charts -f custom-sweep.yaml -n benchmarking
```

## 주요 구성:

* 가변 합성 데이터 분포
* Sweep 모드 (자동): 포화 지점을 발견하기 위해 60초 동안 구성 가능한 요청 수(기본값: 2000)로 시스템을 플러딩
* 포화 주변의 기하학적 클러스터링을 사용한 자동 생성 테스트 단계
* 스트리밍 활성화

## 결과 이해:
도구의 전처리 단계는 60초 동안 2000개의 요청으로 시스템을 플러딩하고 처리 속도를 측정하여 포화를 식별합니다; `saturation_percentile: 95`는 보수적인 추정을 위해 관찰된 속도의 95번째 백분위수를 사용한다는 의미입니다. 로그에서 자동으로 생성된 단계를 검토하고(기하학적 클러스터링은 포화 근처에서 4, 8, 14, 17, 18 QPS와 같이 더 촘촘한 간격을 생성) 감지된 포화 지점을 수동 테스트 기대치와 비교하십시오. 상당한 불일치는 놓쳤을 수 있는 대기열 병목 또는 리소스 제약을 드러내며, 기하학적 분포는 성능이 안정에서 저하로 전환되는 정확한 지점에서 풍부한 데이터를 제공합니다.

**포화 프로브 구성**: sweep 구성의 `numRequests` 매개변수는 초기 포화 발견 단계 동안 전송되는 요청 수를 제어합니다. 기본값 2000은 대부분의 배포에 적합하지만 예상 용량에 따라 조정할 수 있습니다.

<details>
<summary><strong>대안: 원시 Kubernetes YAML</strong></summary>

```yaml
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: inference-perf-sweep
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
      stages: []  # sweep에 의해 자동 생성
      sweep:
        type: geometric
        num_requests: 2000
        timeout: 60
        num_stages: 5
        stage_duration: 180
        saturation_percentile: 95
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
        path: "sweep-test/results"
---
apiVersion: batch/v1
kind: Job
metadata:
  name: inference-perf-sweep
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
          name: inference-perf-sweep
```

</details>
