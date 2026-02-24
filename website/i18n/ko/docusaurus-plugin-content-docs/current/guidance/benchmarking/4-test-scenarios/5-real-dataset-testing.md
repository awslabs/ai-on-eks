---
sidebar_label: 시나리오 5 - 실제 데이터셋 테스트
---

# 시나리오 5: 실제 데이터셋 테스트

## 이 시나리오를 사용해야 하는 경우:
실제 사용자 프롬프트와 쿼리 패턴으로 프로덕션 준비 성능을 검증하기 위해 실제 데이터셋 테스트를 사용하십시오. 이것은 모델이 특정 대화 패턴에 맞게 미세 조정되었을 때, 실제 성능 보장이 있는 모델 버전을 비교할 때, 또는 이해관계자에게 "이것은 이론적 데이터가 아닌 실제 대화에서 어떻게 수행되는지입니다"라고 말해야 할 때 필수적입니다. 분포에 대한 제어가 적어지는 트레이드오프가 있지만, 진정성과 합성 데이터가 놓치는 엣지 케이스를 발견하는 능력을 얻습니다.

## 배포

### Helm 차트 사용 (권장)

```bash
# AI on EKS Helm 저장소 추가
helm repo add ai-on-eks https://awslabs.github.io/ai-on-eks-charts/
helm repo update

# ShareGPT 데이터셋 테스트는 동일한 시나리오를 사용하지만 실제 데이터 사용
helm install sharegpt-test ai-on-eks/benchmark-charts \
  --set benchmark.scenario=baseline \
  --set benchmark.target.baseUrl=http://qwen3-vllm.default:8000 \
  --set benchmark.target.modelName=qwen3-8b \
  --set benchmark.target.tokenizerPath=Qwen/Qwen3-8B \
  --namespace benchmarking --create-namespace

# 자연스러운 대화 복잡성 패턴 모니터링
kubectl logs -n benchmarking -l app.kubernetes.io/component=benchmark -f
```

**참고:** 실제 데이터셋 테스트는 시나리오 1-4와 동일한 부하 패턴을 사용하지만, `data.type: synthetic` 대신 `data.type: shareGPT`를 사용합니다. 모든 시나리오(baseline, saturation, sweep 또는 production)에 실제 데이터를 적용할 수 있습니다.

### 사용자 정의 데이터셋 사용

자체 대화 데이터셋 제공:

```yaml
# custom-dataset.yaml
benchmark:
  scenario: saturation  # 또는 다른 시나리오
  target:
    baseUrl: http://your-model.your-namespace:8000
  # 사용자 정의 데이터셋을 사용하도록 데이터 구성 재정의
  customData:
    enabled: true
    type: custom
    path: /path/to/your/conversations.json
    format: sharegpt  # 또는 openai, alpaca 등
```

사용자 정의 데이터셋의 경우 ConfigMap 또는 PersistentVolume을 사용하여 데이터 파일을 벤치마크 Pod에 마운트해야 합니다.

## 주요 구성:

* ShareGPT 실제 대화 데이터셋 (선택한 시나리오에 따라 다름)
* 모든 부하 패턴 (constant/poisson, 선택한 시나리오에 따라 다름)
* 스트리밍 활성화
* 자연스러운 대화 복잡성 및 길이 분산

## 결과 이해:
실제 대화는 합성 데이터에 없는 자연스러운 복잡성 패턴과 엣지 케이스를 드러냅니다. 느린 처리를 유발하는 문제가 있는 대화 구조나 표현을 노출하는 지연 시간 이상치를 찾으십시오. 유사한 QPS에서 실제 데이터 성능을 합성 테스트와 비교하십시오; 상당한 저하는 실제 대화가 합성 매개변수가 가정한 것보다 더 복잡함을 시사하며, 향후 합성 테스트를 보정하는 데 도움이 됩니다. 다중 턴 대화의 자연스러운 컨텍스트 길이 분산으로 인해 TTFT 변동성이 더 높아지며, 특정 대화 유형에 대한 일관된 오류 패턴은 표적화된 최적화가 필요한 프로덕션 취약점을 드러냅니다. 합성 데이터가 아닌 실제 데이터를 기반으로 이해관계자 약속을 위한 기준으로 이러한 결과를 사용하십시오. "P99 지연 시간은 X입니다"라는 약속을 합성이 아닌 실제 데이터를 기반으로 하십시오.

**중요:** 드리프트를 방지하기 위해 최근 익명화된 프로덕션 샘플로 테스트 데이터셋을 정기적으로 업데이트하십시오. 벤치마크 데이터셋이 6개월 전이지만 사용자 행동이 더 긴 프롬프트로 바뀌었다면 성능 예측이 부정확할 것입니다.

<details>
<summary><strong>대안: 원시 Kubernetes YAML</strong></summary>

```yaml
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: inference-perf-sharegpt
  namespace: benchmarking
data:
  config.yml: |
    api:
      type: completion
      streaming: true

    data:
      type: shareGPT  # 실제 대화 데이터

    load:
      type: constant
      stages:
        - rate: 10
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
        path: "sharegpt-test/results"
---
apiVersion: batch/v1
kind: Job
metadata:
  name: inference-perf-sharegpt
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
          name: inference-perf-sharegpt
```

</details>
