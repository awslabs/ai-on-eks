---
title: Inferentia2의 Mistral-7B
sidebar_position: 2
---
import CollapsibleContent from '@site/src/components/CollapsibleContent';

:::warning
EKS에 ML 모델을 배포하려면 GPU 또는 Neuron 인스턴스에 대한 액세스가 필요합니다. 배포가 작동하지 않는 경우 이러한 리소스에 대한 액세스가 누락되어 있기 때문인 경우가 많습니다. 또한 일부 배포 패턴은 Karpenter 오토스케일링 및 정적 노드 그룹에 의존합니다. 노드가 초기화되지 않으면 Karpenter 또는 노드 그룹의 로그를 확인하여 문제를 해결하십시오.
:::

:::danger

참고: Mistral-7B-Instruct-v0.2는 [Huggingface](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) 리포지토리의 제한 모델입니다. 이 모델을 사용하려면 HuggingFace 토큰이 필요합니다.
HuggingFace에서 토큰을 생성하려면 HuggingFace 계정으로 로그인하고 [설정](https://huggingface.co/settings/tokens) 페이지의 `Access Tokens` 메뉴 항목을 클릭하십시오.

:::

# Inferentia2, Ray Serve, Gradio를 사용한 Mistral-7B-Instruct-v0.2 서빙
이 패턴은 향상된 텍스트 생성 성능을 위해 [AWS Inferentia2](https://aws.amazon.com/ec2/instance-types/inf2/)를 활용하여 Amazon EKS에 [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) 모델 배포를 설명합니다. [Ray Serve](https://docs.ray.io/en/latest/serve/index.html)는 Ray 워커 노드의 효율적인 확장을 보장하고, [Karpenter](https://karpenter.sh/)는 AWS Inferentia2 노드의 프로비저닝을 동적으로 관리합니다. 이 설정은 확장 가능한 클라우드 환경에서 고성능 및 비용 효율적인 텍스트 생성 애플리케이션에 최적화됩니다.

이 패턴을 통해 다음을 수행합니다:

- 동적 노드 프로비저닝을 위한 Karpenter 관리 AWS Inferentia2 노드풀이 있는 [Amazon EKS](https://aws.amazon.com/eks/) 클러스터 생성.
- [trainium-inferentia](https://github.com/awslabs/ai-on-eks/tree/main/infra/trainium-inferentia) Terraform 블루프린트를 사용하여 [KubeRay Operator](https://github.com/ray-project/kuberay) 및 기타 핵심 EKS 애드온 설치.
- 효율적인 확장을 위해 RayServe와 함께 `Mistral-7B-Instruct-v0.2` 모델 배포.

### Mistral-7B-Instruct-v0.2 모델이란?

`mistralai/Mistral-7B-Instruct-v0.2`는 공개적으로 사용 가능한 대화 데이터셋을 사용하여 미세 조정된 `Mistral-7B-v0.2 기본 모델`의 명령어 조정 버전입니다. 명령어를 따르고 작업을 완료하도록 설계되어 챗봇, 가상 어시스턴트 및 작업 지향 대화 시스템과 같은 애플리케이션에 적합합니다. 73억 개의 파라미터를 가진 `Mistral-7B-v0.2` 기본 모델을 기반으로 구축되었으며, 더 빠른 추론을 위한 Grouped-Query Attention (GQA)과 개선된 견고성을 위한 Byte-fallback BPE 토크나이저를 포함한 최신 아키텍처를 사용합니다.

자세한 내용은 [Model Card](https://replicate.com/mistralai/mistral-7b-instruct-v0.2/readme)를 참조하십시오.

## 솔루션 배포
Amazon EKS에서 `Mistral-7B-Instruct-v0.2` 모델을 시작하고 실행해 봅시다! 이 섹션에서는 다음을 다룹니다:

- **사전 요구 사항**: 시작하기 전에 필요한 모든 도구가 설치되어 있는지 확인.
- **인프라 설정**: EKS 클러스터를 생성하고 배포 준비.
- **Ray 클러스터 배포**: 확장성과 효율성을 제공하는 이미지 생성 파이프라인의 핵심.
- **Gradio Web UI 빌드**: Mistral 7B 모델과의 원활한 상호 작용을 위한 사용자 친화적인 인터페이스 생성.

<CollapsibleContent header={<h2><span>사전 요구 사항</span></h2>}>
시작하기 전에 배포 프로세스를 원활하고 문제 없이 진행하기 위해 모든 사전 요구 사항이 갖춰져 있는지 확인하십시오.
머신에 다음 도구가 설치되어 있는지 확인하십시오.

1. [aws cli](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
2. [kubectl](https://Kubernetes.io/docs/tasks/tools/)
3. [terraform](https://learn.hashicorp.com/tutorials/terraform/install-cli)
4. [envsubst](https://pypi.org/project/envsubst/)

<a id="배포"></a>
### 배포

저장소 클론

```bash
git clone https://github.com/awslabs/ai-on-eks.git
```

예제 디렉토리 중 하나로 이동하여 `install.sh` 스크립트 실행

**중요 참고:** 블루프린트를 배포하기 전에 `blueprint.tfvars` 파일에서 리전을 업데이트하십시오.
또한 로컬 리전 설정이 지정된 리전과 일치하는지 확인하여 불일치를 방지하십시오.
예를 들어 `export AWS_DEFAULT_REGION="<REGION>"`을 원하는 리전으로 설정합니다:

```bash
cd ai-on-eks/infra/trainium-inferentia/
./install.sh
```

<a id="리소스-확인"></a>
### 리소스 확인

Amazon EKS 클러스터 확인

```bash
aws eks --region us-west-2 describe-cluster --name trainium-inferentia
```

```bash
# EKS 인증을 위한 k8s 구성 파일 생성
aws eks --region us-west-2 update-kubeconfig --name trainium-inferentia

# 출력은 EKS Managed Node group 노드를 표시합니다
kubectl get nodes
```

</CollapsibleContent>

## Mistral 7B 모델이 있는 Ray 클러스터 배포

`trainium-inferentia` EKS 클러스터가 배포되면 `kubectl`을 사용하여 `/ai-on-eks/blueprints/inference/mistral-7b-rayserve-inf2/` 경로에서 `ray-service-mistral.yaml`을 배포할 수 있습니다.

이 단계에서는 Karpenter 오토스케일링을 사용하는 `x86 CPU` 인스턴스의 `Head Pod` 하나와 [Karpenter](https://karpenter.sh/)에 의해 오토스케일링되는 `inf2.24xlarge` 인스턴스의 `Ray 워커`로 구성된 Ray Serve 클러스터를 배포합니다.

배포를 진행하기 전에 이 배포에서 사용되는 주요 파일을 자세히 살펴보고 기능을 이해해 봅시다:
- **ray_serve_mistral.py:**
  이 스크립트는 AWS Neuron 인프라(Inf2)에서 확장 가능한 모델 서빙을 가능하게 하는 Ray Serve를 사용하여 배포된 두 가지 주요 구성 요소가 있는 FastAPI 애플리케이션을 설정합니다:
  - **mistral-7b Deployment**: 이 클래스는 스케줄러를 사용하여 Mistral 7B 모델을 초기화하고 처리를 위해 Inf2 노드로 이동합니다. 스크립트는 이 Mistral 모델에 대해 grouped-query attention (GQA) 모델에 대한 Transformers Neuron 지원을 활용합니다. `mistral-7b-instruct-v0.2`는 채팅 기반 모델입니다. 스크립트는 실제 프롬프트 주위에 `[INST]` 및 `[/INST]` 토큰을 추가하여 명령어에 필요한 접두사도 추가합니다.
  - **APIIngress**: 이 FastAPI 엔드포인트는 Mistral 7B 모델에 대한 인터페이스 역할을 합니다. 텍스트 프롬프트를 받는 `/infer` 경로에 GET 메서드를 노출합니다. 프롬프트에 텍스트로 응답합니다.

- **ray-service-mistral.yaml:**
  이 RayServe 배포 패턴은 AWS Inferentia2 지원과 함께 Amazon EKS에서 Mistral-7B-Instruct-v0.2 모델을 호스팅하기 위한 확장 가능한 서비스를 설정합니다. 전용 네임스페이스를 생성하고 들어오는 트래픽에 따라 리소스 활용도를 효율적으로 관리하기 위한 오토스케일링 기능이 있는 RayService를 구성합니다. 배포는 RayService 우산 아래에서 제공되는 모델이 수요에 따라 레플리카를 자동으로 조정할 수 있도록 보장하며, 각 레플리카에는 2개의 neuron 코어가 필요합니다. 이 패턴은 성능을 극대화하고 무거운 종속성이 미리 로드되어 시작 지연을 최소화하도록 설계된 사용자 정의 컨테이너 이미지를 사용합니다.

### Mistral-7B-Instruct-v0.2 모델 배포

클러스터가 로컬에서 구성되었는지 확인

```bash
aws eks --region us-west-2 update-kubeconfig --name trainium-inferentia
```

**RayServe 클러스터 배포**

:::info

Mistral-7B-Instruct-v0.2 모델을 배포하려면 Hugging Face Hub 토큰을 환경 변수로 구성하는 것이 필수적입니다. 이 토큰은 인증 및 모델 액세스에 필요합니다. Hugging Face 토큰 생성 및 관리 방법에 대한 지침은 [Hugging Face Token Management](https://huggingface.co/docs/hub/security-tokens)를 참조하십시오.

:::


```bash
# Hugging Face Hub 토큰을 환경 변수로 설정합니다. 이 변수는 ray-service-mistral.yaml 파일을 적용할 때 대체됩니다

export HUGGING_FACE_HUB_TOKEN=$(echo -n "Your-Hugging-Face-Hub-Token-Value" | base64)

cd ai-on-eks/blueprints/inference/mistral-7b-rayserve-inf2
envsubst < ray-service-mistral.yaml| kubectl apply -f -
```

다음 명령을 실행하여 배포 확인

:::info

배포 프로세스는 최대 10분이 소요될 수 있습니다. Head Pod는 2~3분 내에 준비되고, Ray Serve 워커 Pod는 Huggingface에서 이미지 검색 및 모델 배포에 최대 10분이 소요될 수 있습니다.

:::

이 배포는 아래와 같이 `x86` 인스턴스에서 실행되는 Ray head pod와 `inf2.24xl` 인스턴스에서 실행되는 워커 pod를 설정합니다.

```bash
kubectl get pods -n mistral

NAME                                                      READY   STATUS
service-raycluster-68tvp-worker-inf2-worker-group-2kckv   1/1     Running
mistral-service-raycluster-68tvp-head-dmfz5               2/2     Running
```

이 배포는 또한 여러 포트가 구성된 mistral 서비스를 설정합니다. 포트 `8265`는 Ray 대시보드용이고 포트 `8000`은 Mistral 모델 엔드포인트용입니다.

```bash
kubectl get svc -n mistral

NAME                        TYPE       CLUSTER-IP       EXTERNAL-IP   PORT(S)
mistral-service             NodePort   172.20.118.238   <none>        10001:30998/TCP,8000:32437/TCP,52365:31487/TCP,8080:30351/TCP,6379:30392/TCP,8265:30904/TCP
mistral-service-head-svc    NodePort   172.20.245.131   <none>        6379:31478/TCP,8265:31393/TCP,10001:32627/TCP,8000:31251/TCP,52365:31492/TCP,8080:31471/TCP
mistral-service-serve-svc   NodePort   172.20.109.223   <none>        8000:31679/TCP
```

Ray 대시보드의 경우 이러한 포트를 개별적으로 포트 포워딩하여 localhost를 사용하여 로컬에서 웹 UI에 액세스할 수 있습니다.



```bash
kubectl -n mistral port-forward svc/mistral-service 8265:8265
```

`http://localhost:8265`를 통해 웹 UI에 액세스합니다. 이 인터페이스는 Ray 에코시스템 내의 작업 및 액터 배포를 표시합니다.

![RayServe Deployment In Progress](../../img/ray-dashboard-deploying-mistral-inf2.png)

배포가 완료되면 Controller 및 Proxy 상태가 `HEALTHY`이고 Application 상태가 `RUNNING`이어야 합니다

![RayServe Deployment Completed](../../img/ray-dashboard-deployed-mistral-inf2.png)


Ray 대시보드를 사용하여 Serve 배포 및 리소스 활용도를 포함한 Ray 클러스터 배포를 모니터링할 수 있습니다.

![RayServe Cluster](../../img/ray-serve-inf2-mistral-cluster.png)

## Gradio WebUI 앱 배포

[Gradio](https://www.gradio.app/) Web UI는 inf2 인스턴스를 사용하여 EKS 클러스터에 배포된 Mistral7b 추론 서비스와 상호 작용하는 데 사용됩니다.
Gradio UI는 서비스 이름과 포트를 사용하여 포트 `8000`에서 노출되는 mistral 서비스(`mistral-serve-svc.mistral.svc.cluster.local:8000`)와 내부적으로 통신합니다.

Gradio 앱을 위한 기본 Docker(`ai/inference/gradio-ui/Dockerfile-gradio-base`) 이미지를 생성했으며, 이는 모든 모델 추론에 사용할 수 있습니다.
이 이미지는 [Public ECR](https://gallery.ecr.aws/data-on-eks/gradio-web-app-base)에 게시되어 있습니다.

#### Gradio 앱 배포 단계:

다음 YAML 스크립트(`ai/inference/mistral-7b-rayserve-inf2/gradio-ui.yaml`)는 모델 클라이언트 스크립트가 포함된 전용 네임스페이스, 배포, 서비스 및 ConfigMap을 생성합니다.

이를 배포하려면 다음을 실행합니다:

```bash
cd ai-on-eks/blueprints/inference/mistral-7b-rayserve-inf2/
kubectl apply -f gradio-ui.yaml
```

**확인 단계:**
다음 명령을 실행하여 배포, 서비스 및 ConfigMap을 확인합니다:

```bash
kubectl get deployments -n gradio-mistral7b-inf2

kubectl get services -n gradio-mistral7b-inf2

kubectl get configmaps -n gradio-mistral7b-inf2
```

**서비스 포트 포워딩:**

로컬에서 Web UI에 액세스할 수 있도록 포트 포워딩 명령을 실행합니다:

```bash
kubectl port-forward service/gradio-service 7860:7860 -n gradio-mistral7b-inf2
```

#### WebUI 호출

웹 브라우저를 열고 다음 URL로 이동하여 Gradio WebUI에 액세스합니다:

로컬 URL에서 실행 중:  http://localhost:7860

이제 로컬 머신에서 Gradio 애플리케이션과 상호 작용할 수 있습니다.

![Gradio WebUI](../../img/mistral-gradio.png)

#### Mistral 모델과의 상호 작용

`Mistral-7B-Instruct-v0.2` 모델은 채팅 애플리케이션(Q&A, 대화), 텍스트 생성, 지식 검색 등의 용도로 사용할 수 있습니다.

아래 스크린샷은 다양한 텍스트 프롬프트를 기반으로 한 모델 응답의 몇 가지 예를 제공합니다.

![Gradio QA](../../img/mistral-sample-prompt-1.png)

![Gradio Convo 1](../../img/mistral-conv-1.png)

![Gradio Convo 2](../../img/mistral-conv-2.png)

## 정리

마지막으로 더 이상 필요하지 않은 리소스를 정리하고 프로비저닝 해제하는 방법을 안내합니다.

**1단계:** Gradio 앱 및 mistral 추론 배포 삭제

```bash
cd ai-on-eks/blueprints/inference/mistral-7b-rayserve-inf2
kubectl delete -f gradio-ui.yaml
kubectl delete -f ray-service-mistral.yaml
```

**2단계:** EKS 클러스터 정리
이 스크립트는 `-target` 옵션을 사용하여 모든 리소스가 올바른 순서로 삭제되도록 환경을 정리합니다.

```bash
cd ai-on-eks/infra/trainium-inferentia/
./cleanup.sh
```
