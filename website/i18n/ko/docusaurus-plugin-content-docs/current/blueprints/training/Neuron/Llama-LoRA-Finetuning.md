---
sidebar_label: LoRA를 활용한 Llama 3 파인튜닝
---
import CollapsibleContent from '@site/src/components/CollapsibleContent';

:::warning
EKS에서 LLM 파인튜닝을 위해 이 예제를 배포하려면 AWS Trainium EC2 인스턴스에 대한 접근 권한이 필요합니다. 배포가 실패하면 해당 인스턴스 유형에 대한 접근 권한이 있는지 확인하세요. 노드가 시작되지 않으면 Karpenter 또는 노드 그룹 로그를 확인하세요.
:::

:::danger
참고: Llama 3 모델은 Meta 라이선스의 적용을 받습니다. 모델 가중치와 토크나이저를 다운로드하려면 [웹사이트](https://ai.meta.com/)를 방문하여 접근 권한을 요청하기 전에 라이선스에 동의해야 합니다.
:::

:::info
이 블루프린트는 더 나은 관측성, 로깅 및 확장성을 위해 개선 작업 중입니다.
:::

# HuggingFace Optimum Neuron을 사용한 Trn1에서의 Llama 3 파인튜닝

이 가이드는 AWS Trainium (Trn1) EC2 인스턴스를 사용하여 `Llama3-8B` 언어 모델을 파인튜닝하는 방법을 보여줍니다. Neuron과의 쉬운 통합을 위해 HuggingFace Optimum Neuron을 사용합니다.

### Llama 3란?

Llama 3는 텍스트 생성, 요약, 번역, 질의응답과 같은 작업을 위한 대규모 언어 모델(LLM)입니다. 특정 요구사항에 맞게 파인튜닝할 수 있습니다.

### AWS Trainium

AWS Trainium (Trn1) 인스턴스는 고처리량, 저지연 딥러닝을 위해 설계되었으며, Llama 3와 같은 대규모 모델 훈련에 이상적입니다. AWS Neuron SDK는 고급 컴파일러 기술과 혼합 정밀도 훈련으로 모델을 최적화하여 Trainium의 성능을 향상시켜 더 빠르고 정확한 결과를 제공합니다.

## 1. 솔루션 배포

<CollapsibleContent header={<h2><span>사전 요구사항</span></h2>}>
시작하기 전에 필요한 모든 것이 준비되어 있는지 확인하세요:
- 로컬 Mac/Windows 컴퓨터 또는 Amazon EC2 인스턴스를 사용할 수 있습니다.
- Docker를 설치하고(최소 100GB 여유 공간) Docker 이미지가 x86 아키텍처를 사용하는지 확인하세요.
- 다음 도구들을 설치하고 AWS 사용자 또는 역할에 적절한 권한이 있는지 확인하세요:
  * [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
  * [kubectl](https://Kubernetes.io/docs/tasks/tools/)
  * [terraform](https://learn.hashicorp.com/tutorials/terraform/install-cli)
  * [envsubst](https://pypi.org/project/envsubst/)
  * Git (EC2 인스턴스에서만 필요)
  * Python, pip, jq, unzip

**ai-on-eks 저장소 클론:**

사전 요구사항이 준비되면 ai-on-eks GitHub 저장소를 클론합니다:

```bash
git clone https://github.com/awslabs/ai-on-eks.git
```

**trainium-inferentia 디렉토리로 이동:**

```bash
cd ai-on-eks/infra/trainium-inferentia
```

`terraform` 하위 폴더에서 `blueprint.tfvars` 파일에 원하는 AWS 리전을 설정합니다.

**참고:** Trainium 인스턴스는 특정 리전에서만 사용 가능합니다. 지원 리전은 [여기](https://repost.aws/articles/ARmXIF-XS3RO27p0Pd1dVZXQ/what-regions-have-aws-inferentia-and-trainium-instances)에서 확인할 수 있습니다.

`enable_aws_fsx_csi_driver`와 `deploy_fsx_volume` 변수를 `true`로 설정하여 AWS FSx for Lustre CSI 드라이버와 FSx-L 볼륨 프로비저닝을 활성화하세요. 이 파인튜닝 예제에서 사용하지 않는 나머지 리소스는 `false`로 설정할 수 있습니다.

설치 스크립트를 실행하여 필요한 모든 애드온이 포함된 EKS 클러스터를 설정합니다:

```bash
./install.sh
```

<a id="리소스-확인"></a>
### 리소스 확인

이전에 선택한 리전을 사용하여 EKS 클러스터가 실행 중인지 확인합니다:

```bash
aws eks --region AWS_REGION describe-cluster --name trainium-inferentia
```

동일한 리전을 사용하여 EKS와 인증하도록 Kubernetes 설정 파일을 업데이트합니다:

```bash
aws eks --region AWS_REGION update-kubeconfig --name trainium-inferentia

# EKS 관리형 노드 그룹 노드 확인
kubectl get nodes
```

**참고:** AWS_REGION을 이전에 선택한 AWS 리전으로 대체하세요.


</CollapsibleContent>

## 2. Llama 훈련 작업 시작

훈련 스크립트를 시작하기 전에 먼저 유틸리티 파드를 배포합니다. 이 파드의 대화형 셸에 접속하여 파인튜닝 작업의 진행 상황을 모니터링하고, 파인튜닝된 모델 가중치에 접근하며, 샘플 프롬프트에 대해 파인튜닝된 모델이 생성한 출력을 확인할 수 있습니다.

```bash
kubectl apply -f training-artifact-access-pod.yaml
```

훈련 스크립트용 ConfigMap을 생성합니다:

```bash
kubectl apply -f llama3-finetuning-script-configmap.yaml
```

훈련 스크립트가 HuggingFace에서 Llama 3 모델을 다운로드할 수 있으려면 인증 및 모델 접근을 위해 HuggingFace Hub 접근 토큰이 필요합니다. HuggingFace 토큰 생성 및 관리에 대한 자세한 내용은 [Hugging Face Token Management](https://huggingface.co/docs/hub/security-tokens)를 참조하세요.

HuggingFace Hub 토큰을 환경 변수로 설정합니다. `your_huggingface_hub_access_token`을 실제 HuggingFace Hub 접근 토큰으로 대체하세요.

```bash
export HUGGINGFACE_HUB_ACCESS_TOKEN=$(echo -n "your_huggingface_hub_access_token" | base64)
```

다음 명령을 실행하여 Secret과 파인튜닝 Job 리소스를 배포합니다. 이 명령은 yaml을 Kubernetes 클러스터에 적용하기 전에 HUGGINGFACE_HUB_ACCESS_TOKEN 환경 변수를 자동으로 치환합니다.

**참고:** 파인튜닝 컨테이너 이미지는 `us-west-2` ECR 저장소에서 가져옵니다. 이 파인튜닝 예제를 실행하기 위해 선택한 리전에 따라 다른 선호 리전을 제공하는지 확인하려면 [HuggingFace 웹사이트](https://huggingface.co/docs/optimum-neuron/en/containers)를 검토하세요. 다른 지원 리전을 선택하는 경우 아래 명령을 실행하기 전에 lora-finetune-resources.yaml 파일의 컨테이너 이미지 URL에서 AWS 계정 ID와 리전을 업데이트하세요.

```bash
envsubst < lora-finetune-resources.yaml | kubectl apply -f -
```

## 3. 파인튜닝된 Llama3 모델 확인

작업 상태를 확인합니다:

```bash
kubectl get jobs
```

**참고:** 컨테이너가 스케줄링되지 않으면 Karpenter 로그에서 오류를 확인하세요. 선택한 가용 영역(AZ)이나 서브넷에서 사용 가능한 trn1.32xlarge EC2 인스턴스가 없는 경우 이런 현상이 발생할 수 있습니다. 이를 해결하려면 ai-on-eks/infra/base/terraform에 있는 main.tf 파일의 local.azs 필드를 업데이트하세요. 또한 ai-on-eks/infra/base/terraform에 있는 addons.tf 파일의 trainium-trn1 EC2NodeClass가 해당 AZ의 올바른 서브넷을 참조하는지 확인하세요. 그런 다음 ai-on-eks/infra/trainium-inferentia에서 install.sh를 다시 실행하여 Terraform을 통해 변경 사항을 적용합니다.

파인튜닝 작업의 로그를 모니터링하거나, 튜닝된 모델에 접근하거나, 파인튜닝된 모델로 테스트 실행에서 생성된 text-to-SQL 출력을 확인하려면 유틸리티 파드에서 셸을 열고 이러한 항목들이 있는 `/shared` 폴더로 이동합니다. 파인튜닝된 모델은 `llama3_tuned_model_<timestamp>` 이름의 폴더에 저장되며, 샘플 프롬프트에서 생성된 SQL 쿼리는 모델 폴더 옆에 있는 `llama3_finetuning.out` 이름의 로그 파일에서 찾을 수 있습니다.

```bash
kubectl exec -it training-artifact-access-pod -- /bin/bash

cd /shared

ls -l llama3_tuned_model* llama3_finetuning*
```

## 4. 정리

**참고:** 추가 AWS 비용을 피하기 위해 항상 정리 단계를 실행하세요.

이 솔루션에서 생성된 리소스를 제거하려면 ai-on-eks 저장소 루트에서 다음 명령을 실행합니다:

```bash
# Kubernetes 리소스 삭제:
cd blueprints/training/llama-lora-finetuning-trn1
envsubst < lora-finetune-resources.yaml | kubectl delete -f -
kubectl delete -f llama3-finetuning-script-configmap.yaml
kubectl delete -f training-artifact-access-pod.yaml
```

EKS 클러스터 및 관련 리소스 정리:

```bash
cd ../../../infra/trainium-inferentia/terraform
./cleanup.sh
```
