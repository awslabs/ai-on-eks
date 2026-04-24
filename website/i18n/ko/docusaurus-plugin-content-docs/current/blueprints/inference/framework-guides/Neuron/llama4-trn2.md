---
title: Trainium에서 vLLM을 사용한 Llama 4
description: EKS와 Karpenter를 활용하여 AWS Trainium 인스턴스에서 vLLM으로 Llama 4 모델을 배포합니다.
---
import CollapsibleContent from '@site/src/components/CollapsibleContent';

:::danger

Llama 4 모델의 사용은 [Meta Llama 라이선스](https://www.llama.com/llama4/license/)의 적용을 받습니다.
[Hugging Face](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct)를 방문하여 액세스를 요청하기 전에 라이선스에 동의해 주십시오.

:::

# AWS Trainium에서 vLLM을 사용한 Llama 4 추론

이 가이드는 AWS Trainium 인스턴스에서 [optimum-neuron](https://huggingface.co/docs/optimum-neuron/index)과 함께 [vLLM](https://github.com/vllm-project/vllm)을 사용하여 [Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) 모델을 배포하는 방법을 다룹니다.

:::warning[모델 컴파일 필요]

Trainium에서의 Llama 4 추론은 `Llama4NeuronModelForCausalLM` 클래스와 함께 **optimum-neuron >= 0.4.0**을 통해 지원됩니다. 그러나 첫 번째 배포 시 **Neuron 모델 컴파일**이 필요하며, 이는 `vllm serve` 실행 시 자동으로 수행되지만 **30-60분 이상** 소요될 수 있습니다. 모든 구성에 대해 [optimum-neuron-cache](https://huggingface.co/aws-neuron/optimum-neuron-cache)에서 사전 컴파일된 아티팩트를 사용할 수 없을 수 있습니다.

`optimum-cli export neuron` 명령은 Llama 4를 **지원하지 않습니다**. 내부적으로 추론 경로 컴파일을 호출하는 `vllm serve`를 직접 사용하십시오.

:::

## Trainium에서 Llama 4를 사용하는 이유

AWS Trainium은 대용량 HBM 메모리를 제공하여 Llama 4와 같은 대규모 MoE 모델에 탁월한 선택입니다:

| 인스턴스 | 칩 | NeuronCore | HBM 메모리 | Karpenter | EKS Auto Mode |
|----------|------|-------------|------------|-----------|---------------|
| trn1.32xlarge | 16 Trainium v1 | 32 | 512 GiB | 지원 | 지원 |
| trn2.48xlarge | 16 Trainium v2 | 64 | 1.5 TiB | 지원 | 미지원 |

| 장점 | 세부 사항 |
|------|-----------|
| **양자화 불필요** | trn1 (512 GiB)과 trn2 (1.5 TiB) 모두 Scout (~220 GiB)를 네이티브 BF16으로 지원 |
| **Karpenter 자동 프로비저닝** | Neuron NodePool이 워크로드 스케줄링 시 Trainium 노드를 온디맨드로 프로비저닝 |
| **Maverick용 trn2** | trn2.48xlarge (1.5 TiB)는 양자화 없이 BF16으로 Maverick (~800 GiB)을 지원 |

### 메모리 요구 사항

| 모델 | BF16 메모리 | trn1.32xlarge (512 GiB) | trn2.48xlarge (1.5 TiB) |
|------|-------------|-------------------------|-------------------------|
| Scout 17B-16E | ~220 GiB | BF16으로 적합 | BF16으로 적합 |
| Maverick 17B-128E | ~800 GiB | 부적합 | BF16으로 적합 |

:::info

Maverick의 경우, BF16을 위해 충분한 메모리 (1.5 TiB)를 갖춘 `trn2.48xlarge`만 사용 가능합니다. `trn1.32xlarge` (512 GiB)는 메모리가 부족합니다.

:::

:::warning

Trainium 인스턴스 가용성은 리전에 따라 다릅니다. 인프라를 배포하기 전에 [AWS EC2 인스턴스 유형별 리전](https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-regions.html) 페이지에서 현재 가용성을 확인하십시오.

- **trn2.48xlarge**: **EKS Auto Mode에서 지원되지 않습니다** — 추론 전용 클러스터에서 Karpenter를 사용하십시오.

:::

## 모델 컴파일

AWS Neuron DLC는 **optimum-neuron**을 사용하여 Trainium에서 vLLM을 실행합니다. 모델은 서빙 전에 Neuron용으로 사전 컴파일되어야 합니다. DLC는 Hugging Face의 [optimum-neuron-cache](https://huggingface.co/aws-neuron/optimum-neuron-cache)에서 구성(모델, 배치 크기, 시퀀스 길이, 텐서 병렬도, dtype)에 맞는 사전 컴파일된 모델 아티팩트를 확인합니다.

:::info

`optimum-cli export neuron` 명령은 `llama4`를 모델 유형으로 **지원하지 않습니다**. 그러나 `vllm serve`는 `Llama4NeuronModelForCausalLM`을 통한 전체 MoE 지원을 포함하는 별도의 추론 코드 경로(`optimum.neuron.models.inference.llama4`)를 사용합니다. 컴파일은 첫 번째 서빙 시 자동으로 트리거됩니다.

:::

## 소프트웨어 버전

| 구성 요소 | 버전 | 비고 |
|-----------|------|------|
| Neuron SDK | 2.26.1 | 필수 |
| optimum-neuron | >= 0.4.0 | v0.4.0에서 Llama 4 추론 지원 추가 |
| vLLM | 0.11.0 | optimum-neuron Neuron 플랫폼 플러그인 포함 |
| neuronx-distributed | 0.15 | Llama 4 추론에서 사용하는 MoE 모듈 |
| DLC 이미지 | `763104351884.dkr.ecr.<region>.amazonaws.com/huggingface-vllm-inference-neuronx:0.11.0-optimum0.4.5-neuronx-py310-sdk2.26.1-ubuntu22.04` | 최신 버전 |

<CollapsibleContent header={<h2><span>추론 전용 EKS 클러스터 배포</span></h2>}>

이 가이드는 Trainium을 지원하는 기존 EKS 클러스터가 있다고 가정합니다. Karpenter를 사용한 노드 프로비저닝과 사전 구성된 Neuron NodePool이 포함된 [추론 전용 EKS 클러스터](/docs/infra/inference/inference-ready-cluster) 사용을 권장합니다.

### 사전 요구 사항

1. [aws cli](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
2. [kubectl](https://kubernetes.io/docs/tasks/tools/)
3. [Helm 3.0+](https://helm.sh/docs/intro/install/)

### 클러스터 배포

```bash
git clone https://github.com/awslabs/ai-on-eks.git
cd ai-on-eks/infra/solutions/inference-ready-cluster
```

기본 `terraform/blueprint.tfvars`는 Karpenter를 사용합니다 (EKS Auto Mode가 아님). 클러스터는 Trainium 워크로드를 위한 `trn1-neuron`을 포함한 Karpenter NodePool을 생성합니다.

trn2 지원을 추가하려면 `blueprint.tfvars`를 적절한 리전으로 업데이트하고 추가 EC2NodeClass 이름에 `trn2-neuron`을 추가하십시오. trn2 가용성은 [AWS EC2 인스턴스 유형별 리전](https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-regions.html) 페이지에서 확인하십시오.

```hcl
region                              = "<REGION>"  # trn2를 사용할 수 있는 리전 사용
karpenter_additional_ec2nodeclassnames = ["trn2-neuron"]
```

:::note

일부 리전은 가용 영역 수가 적습니다. tfvars에서 `availability_zones_count`를 적절히 설정하십시오 (예: 3개의 AZ가 있는 리전의 경우 `3`).

:::

배포 실행:

```bash
./install.sh
```

### kubectl 구성

```bash
aws eks --region <REGION> update-kubeconfig --name inference-cluster
```

### Karpenter 리소스 확인

```bash
# NodePool 확인
kubectl get nodepools

# EC2NodeClass 확인
kubectl get ec2nodeclasses
```

예상 출력:

```
NAME            NODECLASS       NODES   READY   AGE
trn1-neuron     trn1-neuron     0       True    3m
trn2-neuron     trn2-neuron     0       True    3m
g5-nvidia       g5-nvidia       0       True    3m
...
```

`trn1-neuron` 및 `trn2-neuron` NodePool에는 `aws.amazon.com/neuron` taint가 포함되어 있습니다. Trainium 노드는 일치하는 toleration이 있는 워크로드가 스케줄링될 때 자동으로 프로비저닝됩니다.

### Neuron Device Plugin

Neuron device plugin은 Trainium 워크로드에 **필수**입니다. 추론 전용 클러스터를 사용하는 경우 ArgoCD를 통해 **자동으로 설치**됩니다 ([`aws-neuron.tf`](https://github.com/awslabs/ai-on-eks/blob/main/infra/base/terraform/aws-neuron.tf) 참조). 수동 설치가 필요하지 않습니다.

설치 확인:

```bash
# Neuron device plugin DaemonSet 확인 (Neuron 노드가 프로비저닝되기 전에는 0 desired가 정상)
kubectl get daemonset neuron-device-plugin -n kube-system
```

:::note

ArgoCD 관리 애드온 없이 자체 클러스터를 사용하는 경우 Neuron Helm 차트를 수동으로 설치하십시오:

```bash
kubectl create namespace neuron-healthcheck-system
helm install neuron-helm-chart \
  oci://public.ecr.aws/neuron/neuron-helm-chart \
  --namespace kube-system \
  --version 1.3.0
```

:::

### Neuron 리소스 이름

Trainium 노드가 프로비저닝되면 device plugin은 다음 확장 리소스를 노출합니다:

| 리소스 | 설명 | trn1.32xlarge | trn2.48xlarge |
|--------|------|---------------|---------------|
| `aws.amazon.com/neuron` | Neuron 디바이스 (칩) | 16 | 16 |
| `aws.amazon.com/neuroncore` | NeuronCore (v1 칩당 2개, v2 칩당 4개) | 32 | 64 |

Neuron 디바이스를 할당하려면 Pod 리소스 요청에 `aws.amazon.com/neuron`을 사용하십시오.

</CollapsibleContent>

## Trainium에 Llama 4 Scout 배포

### 1단계: Hugging Face 토큰 Secret 생성

```bash
kubectl create secret generic hf-token --from-literal=token=<your-huggingface-token>
```

### 2단계: Helm으로 배포

**trn2.48xlarge**에서 Scout 배포:

```bash
helm repo add ai-on-eks https://awslabs.github.io/ai-on-eks-charts/
helm repo update

helm install llama4-scout-neuron ai-on-eks/inference-charts \
  --values https://raw.githubusercontent.com/awslabs/ai-on-eks-charts/refs/heads/main/charts/inference-charts/values-llama-4-scout-17b-vllm-neuron.yaml
```

:::info

주요 배포 파라미터:
- **tensor_parallel_size: 16** (NeuronCore가 아닌 Trainium 칩당 하나)
- **Docker 이미지**: 프라이빗 ECR의 AWS Neuron DLC (`763104351884.dkr.ecr.<region>.amazonaws.com/huggingface-vllm-inference-neuronx`)
- **Neuron 디바이스 요청**: 16개 칩 전체를 위한 `aws.amazon.com/neuron: 16`
- **CPU 메모리**: 최소 `384Gi` (가중치 샤딩 시 전체 모델을 CPU 메모리에 로드해야 함)
- **인스턴스 유형**: `trn2.48xlarge` (Scout 및 Maverick 기본값)
- **환경 변수**: 즉석(on-the-fly) Neuron 컴파일을 위해 `VLLM_NEURON_FRAMEWORK=optimum` 필수

:::

### 3단계: 배포 모니터링

배포 후 Karpenter가 자동으로 Trainium 노드를 프로비저닝합니다:

```bash
# 노드 프로비저닝 감시
kubectl get nodeclaims -w

# Pod 상태 확인
kubectl get pods -w
```

배포 중 Pod는 다음 단계를 거칩니다:
1. **Pending** - Trainium 노드 프로비저닝 대기 (~5분)
2. **ContainerCreating** - Neuron DLC 이미지 풀링 (~2.9 GiB)
3. **Running** - Neuron 모델 컴파일 (첫 실행 시 30-60분 이상)
4. **Ready** - vLLM 서버가 요청을 처리 중

:::warning[CPU 메모리 요구 사항]

Pod는 16개 Neuron 디바이스에 걸친 모델 가중치 샤딩을 위해 **최소 384 GiB의 CPU 메모리**가 필요합니다. 메모리가 부족한 경우 (예: 64 GiB) 가중치 로딩 중 OOMKilled됩니다. trn2.48xlarge 인스턴스는 ~2 TiB의 시스템 메모리를 제공하므로 충분한 여유가 있습니다.

:::

:::warning

첫 번째 배포는 Neuron 모델 컴파일로 인해 상당히 오래 걸립니다. 동일한 구성으로의 후속 배포는 캐시된 아티팩트를 사용합니다. 로그에서 컴파일 진행 상황을 모니터링하십시오:

```bash
kubectl logs -f -l app.kubernetes.io/instance=llama4-scout-neuron
```

:::

**trn2.48xlarge에서 테스트된 배포 타임라인 (Scout):**

| 단계 | 소요 시간 | 설명 |
|------|-----------|------|
| 노드 프로비저닝 | ~5분 | Karpenter가 trn2.48xlarge 프로비저닝 |
| 이미지 풀 | ~30초 | DLC 이미지 (~2.9 GiB, 첫 풀 이후 캐시됨) |
| HLO 생성 | ~60초 | context_encoding 및 token_generation용 HLO 생성 |
| Neuron 컴파일 | ~200초 | neuronx-cc가 HLO를 NEFF로 컴파일 (target=trn2) |
| 모델 빌드 | ~650초 | 가중치 레이아웃 변환 |
| 가중치 로딩 | ~5분 | 16개 Neuron 디바이스에 가중치 다운로드, 샤딩, 로드 |
| **총 소요 (첫 배포)** | **~20분** | 후속 배포는 캐시된 컴파일 아티팩트 재사용 |

완료되면 vLLM 서버가 시작됩니다:

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Trainium2에 Llama 4 Maverick 배포

Maverick은 `trn2.48xlarge` (1.5 TiB HBM)가 필요하며 양자화 없이 네이티브 BF16으로 실행됩니다. 클러스터에 `trn2-neuron` Karpenter NodePool이 구성되어 있는지 확인하십시오 (위의 클러스터 설정 참조).

:::info

수동 모델 컴파일이 필요하지 않습니다. Scout와 마찬가지로 `vllm serve`가 첫 번째 시작 시 optimum-neuron을 통해 자동으로 JIT 컴파일을 트리거합니다. Kubernetes가 Pod를 재시작하지 않도록 컴파일이 완료될 때까지 충분한 시작 시간(liveness/readiness probe의 `initialDelaySeconds`)이 구성되어 있는지 확인하십시오.

:::

```bash
helm install llama4-maverick-neuron ai-on-eks/inference-charts \
  --values https://raw.githubusercontent.com/awslabs/ai-on-eks-charts/refs/heads/main/charts/inference-charts/values-llama-4-maverick-17b-vllm-neuron.yaml
```

:::warning

- `trn2.48xlarge` 가용성은 제한적입니다. 배포 전에 [AWS EC2 인스턴스 유형별 리전](https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-regions.html)을 확인하십시오.
- AWS 계정에 Trainium 인스턴스에 대한 충분한 [서비스 할당량](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-resource-limits.html)이 있는지 확인하십시오 (Maverick은 192 vCPU가 필요합니다).

:::

## 컴파일 캐시 유지

기본적으로 Neuron 컴파일 아티팩트는 임시 컨테이너 스토리지(`/var/tmp/neuron-compile-cache/`)에 저장됩니다. 이는 **Pod 재시작 시마다 재컴파일이 발생**하여 시작 시간에 ~20분이 추가됨을 의미합니다. 프로덕션 배포의 경우 다음 방법 중 하나로 캐시를 유지하십시오:

### 옵션 1: S3 기반 캐시 (권장)

`NEURON_COMPILE_CACHE_URL` 환경 변수를 설정하여 컴파일된 아티팩트를 S3에 저장합니다:

```yaml
env:
  - name: NEURON_COMPILE_CACHE_URL
    value: "s3://your-bucket/neuron-compile-cache/"
```

이를 통해 모든 Pod (교체 Pod 및 스케일아웃 복제본 포함)가 동일한 컴파일 캐시를 공유할 수 있습니다.

### 옵션 2: PersistentVolume 마운트

컴파일 캐시 디렉토리에 PersistentVolume을 마운트합니다:

```yaml
volumeMounts:
  - name: neuron-cache
    mountPath: /var/tmp/neuron-compile-cache
volumes:
  - name: neuron-cache
    persistentVolumeClaim:
      claimName: neuron-compile-cache-pvc
```

:::info

Hugging Face의 [optimum-neuron-cache](https://huggingface.co/aws-neuron/optimum-neuron-cache)는 로컬 컴파일 전에 자동으로 확인됩니다. 정확한 구성(모델, 배치 크기, 시퀀스 길이, 텐서 병렬도, dtype)에 대한 사전 컴파일된 아티팩트가 있는 경우 재컴파일 대신 다운로드됩니다. Llama 4 구성이 캐시에 추가됨에 따라 콜드 스타트 시간이 개선될 것입니다.

:::

## 모델 테스트

### 포트 포워딩

```bash
kubectl port-forward svc/llama4-scout-neuron 8000:8000
```

### 채팅 완성 요청

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "messages": [
      {"role": "user", "content": "대규모 언어 모델에서 Mixture of Experts 아키텍처의 장점을 설명해 주세요."}
    ],
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

### 사용 가능한 모델 목록 확인

```bash
curl http://localhost:8000/v1/models | python3 -m json.tool
```

### 멀티모달 요청 (텍스트 + 이미지)

Llama 4는 멀티모달 추론을 지원합니다. 텍스트와 함께 이미지 URL을 전송할 수 있습니다:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "이 이미지에서 무엇이 보이는지 설명해 주세요."},
          {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"}}
        ]
      }
    ],
    "max_tokens": 256
  }'
```

## Open WebUI 배포

[Open WebUI](https://github.com/open-webui/open-webui)는 모델과 상호 작용하기 위한 ChatGPT 스타일의 인터페이스를 제공합니다.

```bash
helm repo add open-webui https://helm.openwebui.com/
helm repo update

helm install open-webui open-webui/open-webui \
  --namespace open-webui --create-namespace \
  --set ollama.enabled=false \
  --set env.OPENAI_API_BASE_URL=http://llama4-scout-neuron.default.svc.cluster.local:8000/v1 \
  --set env.OPENAI_API_KEY=dummy
```

UI에 접속:

```bash
kubectl port-forward svc/open-webui 8080:80 -n open-webui
```

브라우저에서 [http://localhost:8080](http://localhost:8080)을 열고 새 계정을 등록하십시오. 모델 선택기에 모델이 표시됩니다.

## 모니터링

### 추론 로그 확인

```bash
# vLLM Neuron 로그 확인
kubectl logs -l app.kubernetes.io/instance=llama4-scout-neuron --tail=100

# 토큰 생성 처리량 모니터링
kubectl logs -l app.kubernetes.io/instance=llama4-scout-neuron -f | grep "tokens/s"
```

### 관측성 대시보드

클러스터에서 관측성 스택이 활성화된 경우 Grafana에 접속할 수 있습니다:

```bash
kubectl port-forward svc/kube-prometheus-stack-grafana 3000:80 -n monitoring
```

## 정리

모델 배포 제거:

```bash
# Scout 제거
helm uninstall llama4-scout-neuron

# Maverick 제거 (배포한 경우)
helm uninstall llama4-maverick-neuron
```

전체 클러스터 인프라를 삭제하려면:

```bash
cd ai-on-eks/infra/solutions/inference-ready-cluster
./cleanup.sh
```
