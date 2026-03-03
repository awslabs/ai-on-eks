---
sidebar_label: 리소스
---

# 리소스

## Helm 차트 저장소

공식 벤치마크 차트는 [AI on EKS Charts 저장소](https://github.com/awslabs/ai-on-eks-charts/tree/main/charts/benchmark-charts)에서 유지 관리됩니다. 이 저장소에는 다음이 포함되어 있습니다:

- **values.yaml**: 사용 가능한 모든 옵션이 포함된 완전한 구성 참조
- **templates/**: job, configmap 및 서비스 계정을 위한 Kubernetes 리소스 템플릿
- **scenarios/**: 사전 구성된 시나리오 정의 (baseline, saturation, sweep, production)
- **README.md**: 자세한 사용 지침 및 예제

### values.yaml을 통한 사용자 정의

기본값을 재정의하기 위한 사용자 정의 values 파일 생성:

```yaml
# custom-benchmark.yaml
benchmark:
  scenario: saturation
  target:
    baseUrl: http://your-model.your-namespace:8000
    modelName: your-model-name

  # 시나리오별 설정 재정의
  scenarios:
    saturation:
      load:
        stages:
          - rate: 10
            duration: 300
          - rate: 50
            duration: 300

  # 리소스 할당
  resources:
    requests:
      cpu: "4"
      memory: "8Gi"

  # Pod 어피니티 사용자 정의
  affinity:
    enabled: true
    targetLabels:
      app: your-inference-service
```

사용자 정의 값으로 배포:
```bash
helm install my-benchmark ai-on-eks/benchmark-charts -f custom-benchmark.yaml -n benchmarking
```

## 대안: SentencePiece가 포함된 사용자 정의 컨테이너

Helm 차트 외부의 사용자 정의 배포의 경우 종속성이 사전 설치된 컨테이너 이미지를 빌드할 수 있습니다:

```bash
# 사용자 정의 Dockerfile 생성
cat > Dockerfile <<'EOF'
FROM quay.io/inference-perf/inference-perf:v0.2.0

# sentencepiece 설치
RUN pip install --no-cache-dir sentencepiece protobuf

USER 1000
EOF

# 레지스트리에 빌드 및 푸시
docker build -t <your-registry>/inference-perf:v0.2.0-sentencepiece .
docker push <your-registry>/inference-perf:v0.2.0-sentencepiece

# 새 이미지를 사용하도록 Job 업데이트
kubectl patch job inference-perf-run -n benchmarking \
  --type='json' \
  -p='[{"op": "replace", "path": "/spec/template/spec/containers/0/image", "value":"<your-registry>/inference-perf:v0.2.0-sentencepiece"}]'
```

## 대안: 완전한 Kubernetes 매니페스트

수동 배포 또는 교육 목적으로 런타임 종속성 설치가 포함된 완전한 YAML 매니페스트입니다:

```bash
cat > inference-perf-fixed.yaml <<'EOF'
---
apiVersion: v1
kind: Namespace
metadata:
  name: benchmarking
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: inference-perf-sa
  namespace: benchmarking
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: inference-perf-config
  namespace: benchmarking
data:
  config.yml: |
    load_generator:
      concurrency: 10
      duration: 60

    model:
      model_name: qwen3-8b
      base_url: http://qwen3-vllm.default:8000
      ignore_eos: true

      tokenizer:
        pretrained_model_name_or_path: Qwen/Qwen3-8B

    storage:
      simple_storage_service:
        bucket_name: "inference-perf-results"
        path: "inference-perf/results"
---
apiVersion: batch/v1
kind: Job
metadata:
  name: inference-perf-run
  namespace: benchmarking
  labels:
    app: inference-perf
spec:
  backoffLimit: 2
  ttlSecondsAfterFinished: 3600
  template:
    metadata:
      labels:
        app: inference-perf
    spec:
      restartPolicy: Never
      serviceAccountName: inference-perf-sa

      # 재현 가능한 결과를 위해 추론 Pod와 동일 AZ 배치
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
            echo "Installing dependencies..."
            pip install --no-cache-dir sentencepiece==0.2.0 protobuf==5
            echo "Dependencies installed successfully"
            echo "Starting inference-perf..."
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
            name: inference-perf-config
EOF
```

## 실행

```bash
kubectl apply -f inference-perf-fixed.yaml
```
