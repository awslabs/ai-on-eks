---
sidebar_label: 완전한 배포 예제
---

# 완전한 배포 예제

이 예제는 S3 스토리지, 현실적인 부하 테스트 및 적절한 AWS 통합을 갖춘 프로덕션 준비 배포를 보여줍니다. 벤치마크를 배포하려면 다음 단계를 따르십시오.

## 단계 0: 환경 설정 (선택 사항)

배포 경로를 선택하십시오:

### 경로 A (권장): ai-on-eks 블루프린트 사용

* kube-prometheus-stack을 자동으로 배포합니다
* 고정된 Prometheus URL: http://kube-prometheus-stack-prometheus.monitoring:9090
* 팔로우: https://awslabs.github.io/ai-on-eks/docs/infra/inference-ready-cluster

### 경로 B: 기존 EKS 클러스터

기존 클러스터가 있는 경우 다음 사전 요구 사항을 확인하십시오:

* Kubernetes 1.28+ 이상의 EKS 클러스터
* NVIDIA 드라이버가 설치된 GPU 노드 (g5.xlarge 이상)
* Karpenter (선택 사항이지만 오토스케일링에 권장)
* S3 접근을 위해 구성된 Pod Identity 또는 IRSA
* 클러스터 접근이 구성된 kubectl
* Prometheus가 미리 배포되어 있어야 함
* 메트릭 수집은 벤치마킹에 선택 사항
* Prometheus 서비스 이름과 네임스페이스를 알아야 함
* 예: http://&lt;your-prometheus-service&gt;.&lt;namespace&gt;:9090

## 단계 1: 추론 모델 배포

벤치마크를 실행하기 전에 활성 LLM 추론 엔드포인트가 필요합니다.

**경로 A 사용자:** 사전 구성된 추론 배포와 함께 ai-on-eks 블루프린트를 사용하여 배포한 경우 단계 2로 건너뛰십시오.

**경로 B 사용자:** inference-charts를 사용하여 선택한 모델로 vLLM을 배포합니다:

```bash
# AI on EKS Helm 저장소 추가
helm repo add ai-on-eks https://awslabs.github.io/ai-on-eks-charts/
helm repo update

# vLLM으로 Qwen3-8B 배포
helm install qwen3-vllm ai-on-eks/inference-charts \
  --set model=Qwen/Qwen3-8B \
  --set inference.framework=vllm \
  --namespace default --create-namespace

# 배포 확인
kubectl get pods -n default -l app.kubernetes.io/name=inference-charts
kubectl logs -n default -l app.kubernetes.io/name=inference-charts -f
```

벤치마킹을 진행하기 전에 모델이 준비될 때까지 기다리십시오. 이는 모델 크기와 다운로드 속도에 따라 일반적으로 3-10분이 소요됩니다.

## 단계 2: AWS 스토리지 설정 (S3 사용 - 권장)

**참고:** ai-on-eks inference-ready-cluster 블루프린트를 사용하여 클러스터를 배포한 경우 EKS Pod Identity Agent 애드온이 이미 설치되어 있습니다. 아래의 애드온 설치 명령을 건너뛰고 S3 버킷 및 IAM 역할 생성으로 직접 진행할 수 있습니다.

하드코딩된 자격 증명 없이 벤치마크 Pod가 S3에 결과를 쓸 수 있도록 AWS 자격 증명을 설정합니다.

```bash
# 벤치마크 결과를 위한 S3 버킷 생성
export BUCKET_NAME="inference-perf-results-$(aws sts get-caller-identity --query Account --output text)"
aws s3 mb s3://${BUCKET_NAME} --region us-west-2

# EKS Pod Identity Agent 설치 (블루프린트 참조에 이미 배포됨 - https://awslabs.github.io/ai-on-eks/docs/infra/inference-ready-cluster)

aws eks create-addon \
  --cluster-name my-cluster \
  --addon-name eks-pod-identity-agent \
  --addon-version v1.3.0-eksbuild.1

# S3 권한이 있는 IAM 역할 생성

cat > trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {
      "Service": "pods.eks.amazonaws.com"
    },
    "Action": [
      "sts:AssumeRole",
      "sts:TagSession"
    ]
  }]
}
EOF


aws iam create-role \
  --role-name InferencePerfRole \
  --assume-role-policy-document file://trust-policy.json



aws iam attach-role-policy \
  --role-name InferencePerfRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess



# Kubernetes 서비스 계정에 역할 연결

aws eks create-pod-identity-association \
  --cluster-name my-cluster \
  --namespace benchmarking \
  --service-account inference-perf-sa \
  --role-arn arn:aws:iam::ACCOUNT_ID:role/InferencePerfRole
```

## 단계 3: 벤치마크 리소스 배포

### 옵션 A: Helm 차트 사용 (권장)

[AI on EKS Benchmark Helm Chart](https://github.com/awslabs/ai-on-eks-charts/tree/main/charts/benchmark-charts)는 간소화된 구성 관리와 함께 프로덕션 준비 배포를 제공합니다.

**벤치마크 설치:**

```bash
# AI on EKS Helm 저장소 추가
helm repo add ai-on-eks https://awslabs.github.io/ai-on-eks-charts/
helm repo update

# 프로덕션 시뮬레이션 테스트 배포
helm install production-test ai-on-eks/benchmark-charts \
  --set benchmark.scenario=production \
  --set benchmark.target.baseUrl=http://qwen3-vllm.default:8000 \
  --set benchmark.target.modelName=qwen3-8b \
  --set benchmark.target.tokenizerPath=Qwen/Qwen3-8B \
  --namespace benchmarking --create-namespace
```

**사용자 정의 값으로 커스터마이징:**

```yaml
# custom-benchmark.yaml
benchmark:
  scenario: production
  target:
    baseUrl: http://qwen3-vllm.default:8000
    modelName: qwen3-8b
    tokenizerPath: Qwen/Qwen3-8B

  # S3 스토리지 구성
  storage:
    s3:
      enabled: true
      bucketName: inference-perf-results
      path: "inference-perf/results"

  # 동일 AZ 배치를 위한 Pod 어피니티
  affinity:
    enabled: true
    targetLabels:
      app.kubernetes.io/component: qwen3-vllm

  # 리소스 할당
  resources:
    requests:
      cpu: "2"
      memory: "4Gi"
    limits:
      cpu: "4"
      memory: "8Gi"
```

사용자 정의 값으로 배포:
```bash
helm install production-test ai-on-eks/benchmark-charts \
  -f custom-benchmark.yaml \
  --namespace benchmarking --create-namespace
```

**Helm 접근 방식의 이점:**
- 장황한 YAML 대신 values.yaml을 통한 **간소화된 구성**
- **사전 구성된 시나리오** (baseline, saturation, sweep, production)
- Pod 어피니티, 리소스 및 종속성에 대한 **일관된 기본값**
- Helm 버전 관리를 통한 **손쉬운 업그레이드** 및 롤백

### 옵션 B: 수동 Kubernetes YAML (교육용)

학습 목적이나 고도로 사용자 정의된 배포의 경우 Kubernetes 매니페스트로 직접 배포할 수 있습니다. 이 접근 방식은 모든 리소스에 대한 완전한 투명성을 제공합니다.

<details>
<summary><strong>펼쳐서 보기: 수동 YAML 배포 지침</strong></summary>

#### 모델 종속성 처리

일부 모델은 기본 inference-perf 컨테이너에 포함되지 않은 추가 Python 패키지를 필요로 합니다. 예를 들어, Mistral 및 Llama 모델에는 `sentencepiece`가 필요합니다. Qwen3 모델은 이미 포함된 tiktoken을 사용하므로 추가 패키지가 필요하지 않습니다.

**두 가지 접근 방식:**

#### 접근 방식 A: 런타임 설치 (권장 - 간단)
벤치마크 실행 전 메인 컨테이너 시작의 일부로 종속성을 설치합니다:

```yaml
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

      ...

      containers:
      - name: inference-perf
        image: quay.io/inference-perf/inference-perf:v0.2.0
        command: ["/bin/sh", "-c"]
        args:
          - |
            echo "Installing dependencies..."
            pip install --no-cache-dir sentencepiece==0.2.0 protobuf==5.29.2
            echo "Dependencies installed successfully"
            echo "Starting inference-perf..."
            inference-perf --config_file /workspace/config.yml

```

#### 접근 방식 B: 사용자 정의 컨테이너 이미지 (고급)
종속성이 사전 설치된 사용자 정의 이미지를 빌드합니다:

```dockerfile
FROM quay.io/inference-perf/inference-perf:v0.2.0

RUN pip install --no-cache-dir sentencepiece==0.2.0 protobuf==5.29.2
```

#### 각 접근 방식을 사용해야 하는 경우:

* 빠른 테스트와 유연성을 위해 **접근 방식 A** 사용
* 프로덕션 재현성과 더 빠른 시작을 위해 **접근 방식 B** 사용

### 네임스페이스 및 서비스 계정 생성

```bash
cat <<EOF | kubectl apply -f -
# 벤치마크 워크로드를 위한 네임스페이스
apiVersion: v1
kind: Namespace
metadata:
  name: benchmarking

---
# 서비스 계정 (Pod Identity를 통해 AWS IAM에 연결)
apiVersion: v1
kind: ServiceAccount
metadata:
  name: inference-perf-sa
  namespace: benchmarking
EOF
```



### HuggingFace 토큰 Secret 생성 (선택 사항이지만 권장)

모델이 HuggingFace에서 토크나이저를 다운로드하기 위해 인증이 필요한 경우 아래 명령을 사용하여 secret을 생성합니다. 이 접근 방식은 실수로 버전 관리에 커밋될 수 있는 YAML 파일에 secret을 정의하는 것보다 더 안전합니다.

**단계 1: HuggingFace 토큰 얻기**

* https://huggingface.co/settings/tokens 으로 이동
* 읽기 토큰이 없는 경우 생성

**단계 2: secret 생성**

```bash
kubectl create secret generic hf-token \
  --from-literal=token=YOUR_HUGGINGFACE_TOKEN_HERE \
  --namespace=benchmarking
```


**단계 3: secret이 생성되었는지 확인**

```bash
kubectl get secret hf-token -n benchmarking
```


**보안 참고:** Git 저장소에 secret을 커밋하지 마십시오. 프로덕션 배포에는 항상 명령형 명령 또는 외부 secret 관리 도구(AWS Secrets Manager, HashiCorp Vault 등)를 사용하십시오.

### ConfigMap 및 Job 생성

```bash
cat <<EOF | kubectl apply -f -
# 벤치마크 구성
apiVersion: v1
kind: ConfigMap
metadata:
  name: inference-perf-config
  namespace: benchmarking
data:
  config.yml: |
    # API 구성
    api:
      type: completion
      streaming: true
    # 데이터 생성 - 현실적인 분포를 가진 합성
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
    # 부하 패턴 - 5분 동안 10 QPS의 Poisson 도착
    load:
      type: poisson
      stages:
        - rate: 10
          duration: 300
      num_workers: 4
    # 모델 서버
    server:
      type: vllm
      model_name: qwen3-8b
      base_url: http://qwen3-vllm.default:8000
      ignore_eos: true

    # 토크나이저
    tokenizer:
      pretrained_model_name_or_path: Qwen/Qwen3-8B

    # 스토리지 - 결과가 S3에 자동으로 저장됩니다
    storage:
      simple_storage_service:
        bucket_name: "inference-perf-results"
        path: "inference-perf/results"
    # 선택 사항: Prometheus 메트릭 수집
    # metrics:
    #   type: prometheus
    #   prometheus:
    #     url: http://kube-prometheus-stack-prometheus.monitoring:9090
    #     scrape_interval: 15
EOF
---

cat <<EOF | kubectl apply -f -
# 벤치마크 Job
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
            echo "Starting inference-perf..."
            inference-perf --config_file /workspace/config.yml
        volumeMounts:
          - name: config
            mountPath: /workspace/config.yml
            subPath: config.yml
        env:
          - name: HF_TOKEN
            valueFrom:
              secretKeyRef:
                name: hf-token
                key: token
                optional: true
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

**팁:** 표시된 리소스 값은 시작점입니다. 더 높은 동시성 수준이나 더 긴 테스트 기간의 경우 `kubectl top pod -n benchmarking`으로 Pod 리소스 사용량을 모니터링하고 적절히 조정하십시오.

</details>

---

## 단계 4: 배포 및 모니터링

### Helm 배포의 경우:

```bash
# Job 진행 상황 모니터링
kubectl get jobs -n benchmarking -w

# 벤치마크 진행 상황을 보기 위해 로그 팔로우
kubectl logs -n benchmarking -l app.kubernetes.io/component=benchmark -f

# Helm 릴리스 상태 확인
helm status production-test -n benchmarking
```

### 수동 YAML 배포의 경우:

위의 옵션 B에서 매니페스트를 `inference-perf-complete.yaml`로 저장하고 배포합니다:

```bash
# 모든 리소스 배포
kubectl apply -f inference-perf-complete.yaml

# Job 진행 상황 모니터링
kubectl get jobs -n benchmarking -w

# 벤치마크 진행 상황을 보기 위해 로그 팔로우
kubectl logs -n benchmarking -l app=inference-perf -f
```

## 단계 5: 결과 검색

### S3 스토리지 사용 (권장):
결과가 S3 버킷에 자동으로 업로드됩니다. 직접 접근합니다:

```bash
# S3에서 결과 나열 (단계 2의 버킷 이름 사용)
aws s3 ls s3://${BUCKET_NAME}/inference-perf/ --recursive

# 특정 보고서 다운로드
aws s3 cp s3://${BUCKET_NAME}/inference-perf/20251020-143000/summary_lifecycle_metrics.json ./
```

### 로컬 스토리지 사용 (대안):
S3 대신 `local_storage`를 사용하는 경우 Pod가 종료되기 전에 수동으로 결과를 복사해야 합니다:

```bash
# config.yml에서 다음을 사용합니다:

storage:

  local_storage:

    path: "reports-results"



# Pod 이름 가져오기

POD_NAME=$(kubectl get pods -n benchmarking -l app=inference-perf -o jsonpath='{.items[0].metadata.name}')



# Pod에서 결과 복사

kubectl cp benchmarking/$POD_NAME:/reports-* ./local-reports/
```

### 스토리지 비교:

| 기능 | 로컬 스토리지 | S3 스토리지 |
|---|---|---|
| 설정 | 필요 없음 | AWS 자격 증명 필요 |
| 지속성 | 수동 복사 필요 | 자동 |
| 적합한 용도 | 빠른 테스트, 실험 | 프로덕션, 자동화 |
| 결과 접근 | kubectl cp 명령 | AWS S3 명령/콘솔 |
