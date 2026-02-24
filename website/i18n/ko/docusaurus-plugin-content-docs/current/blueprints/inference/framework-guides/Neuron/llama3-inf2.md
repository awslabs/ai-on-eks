---
title: Inferentia2의 Llama-3-8B
sidebar_position: 3
description: AWS Inferentia 가속기에서 효율적인 추론을 위해 Llama-3 모델 서빙.
---
import CollapsibleContent from '@site/src/components/CollapsibleContent';

:::warning
EKS에 ML 모델을 배포하려면 GPU 또는 Neuron 인스턴스에 대한 액세스가 필요합니다. 배포가 작동하지 않는 경우 이러한 리소스에 대한 액세스가 누락되어 있기 때문인 경우가 많습니다. 또한 일부 배포 패턴은 Karpenter 오토스케일링 및 정적 노드 그룹에 의존합니다. 노드가 초기화되지 않으면 Karpenter 또는 노드 그룹의 로그를 확인하여 문제를 해결하십시오.
:::


:::danger

참고: 이 Llama-3 Instruct 모델의 사용은 Meta 라이선스의 적용을 받습니다.
모델 가중치와 토크나이저를 다운로드하려면 [웹사이트](https://huggingface.co/meta-llama/Meta-Llama-3-8B)를 방문하여 액세스를 요청하기 전에 라이선스에 동의해 주십시오.

:::

:::info

관측성, 로깅 및 확장성 측면의 개선 사항을 포함하기 위해 이 블루프린트를 적극적으로 개선하고 있습니다.
:::


# Inferentia, Ray Serve 및 Gradio를 사용한 Llama-3-8B Instruct 모델 서빙

[Ray Serve](https://docs.ray.io/en/latest/serve/index.html)를 사용하여 Amazon Elastic Kubernetes Service (EKS)에 [Meta Llama-3-8B Instruct](https://ai.meta.com/llama/#inside-the-model) 모델을 배포하는 포괄적인 가이드에 오신 것을 환영합니다.

이 튜토리얼에서는 Llama-3의 강력한 기능을 활용하는 방법뿐만 아니라 대규모 언어 모델(LLM)을 효율적으로 배포하는 복잡한 과정에 대한 통찰력을 얻을 수 있습니다. 특히 대규모 언어 모델 배포 및 확장에 최적화된 `inf2.24xlarge` 및 `inf2.48xlarge`와 같은 [trn1/inf2](https://aws.amazon.com/machine-learning/neuron/) (AWS Trainium 및 Inferentia 기반) 인스턴스에서의 배포를 다룹니다.

### Llama-3-8B Instruct란?

Meta는 8B 및 70B 크기의 사전 훈련 및 명령어 조정 생성 텍스트 모델 컬렉션인 Meta Llama 3 대규모 언어 모델(LLM) 제품군을 개발하고 출시했습니다. Llama 3 명령어 조정 모델은 대화 사용 사례에 최적화되어 있으며 일반적인 업계 벤치마크에서 사용 가능한 많은 오픈 소스 채팅 모델을 능가합니다. 또한 이러한 모델을 개발할 때 유용성과 안전성을 최적화하는 데 세심한 주의를 기울였습니다.

Llama3 크기 및 모델 아키텍처에 대한 자세한 정보는 [여기](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)에서 확인할 수 있습니다.

**확장성 및 가용성**

Llama-3와 같은 대규모 언어 모델(`LLM`)을 배포할 때 주요 과제 중 하나는 적절한 하드웨어의 확장성과 가용성입니다. 기존 `GPU` 인스턴스는 높은 수요로 인해 부족한 경우가 많아 리소스를 효과적으로 프로비저닝하고 확장하기가 어렵습니다.

반면 `trn1.32xlarge`, `trn1n.32xlarge`, `inf2.24xlarge` 및 `inf2.48xlarge`와 같은 `Trn1/Inf2` 인스턴스는 LLM을 포함한 생성형 AI 모델의 고성능 딥러닝(DL) 훈련 및 추론을 위해 특별히 구축되었습니다. 확장성과 가용성을 모두 제공하여 리소스 병목 현상이나 지연 없이 필요에 따라 `Llama-3` 모델을 배포하고 확장할 수 있습니다.

**비용 최적화**

기존 GPU 인스턴스에서 LLM을 실행하면 GPU의 부족과 경쟁적인 가격으로 인해 비용이 많이 들 수 있습니다. **Trn1/Inf2** 인스턴스는 비용 효율적인 대안을 제공합니다. AI 및 기계 학습 작업에 최적화된 전용 하드웨어를 제공함으로써 Trn1/Inf2 인스턴스를 통해 비용의 일부로 최고 수준의 성능을 달성할 수 있습니다. 이러한 비용 최적화를 통해 예산을 효율적으로 할당하여 LLM 배포를 접근 가능하고 지속 가능하게 만들 수 있습니다.

**성능 향상**

Llama-3는 GPU에서 고성능 추론을 달성할 수 있지만, Neuron 가속기는 성능을 한 단계 더 끌어올립니다. Neuron 가속기는 기계 학습 워크로드를 위해 특별히 구축되어 Llama-3의 추론 속도를 크게 향상시키는 하드웨어 가속을 제공합니다. 이는 Trn1/Inf2 인스턴스에 Llama-3를 배포할 때 더 빠른 응답 시간과 개선된 사용자 경험으로 이어집니다.


### 예제 사용 사례

회사가 고객 지원을 제공하기 위해 Llama-3 챗봇을 배포하려고 합니다. 회사는 대규모 고객 기반을 보유하고 있으며 피크 시간에 많은 양의 채팅 요청을 받을 것으로 예상합니다. 회사는 높은 요청량을 처리하고 빠른 응답 시간을 제공할 수 있는 인프라를 설계해야 합니다.

회사는 Inferentia2 인스턴스를 사용하여 Llama-3 챗봇을 효율적으로 확장할 수 있습니다. Inferentia2 인스턴스는 기계 학습 작업을 위한 특수 하드웨어 가속기입니다. 기계 학습 워크로드에 대해 GPU보다 최대 20배 더 나은 성능과 최대 7배 더 낮은 비용을 제공할 수 있습니다.

회사는 또한 Ray Serve를 사용하여 Llama-3 챗봇을 수평으로 확장할 수 있습니다. Ray Serve는 기계 학습 모델을 서빙하기 위한 분산 프레임워크입니다. 수요에 따라 모델을 자동으로 확장하거나 축소할 수 있습니다.

Llama-3 챗봇을 확장하기 위해 회사는 여러 Inferentia2 인스턴스를 배포하고 Ray Serve를 사용하여 인스턴스 간에 트래픽을 분산할 수 있습니다. 이를 통해 회사는 높은 요청량을 처리하고 빠른 응답 시간을 제공할 수 있습니다.

## 솔루션 아키텍처

이 섹션에서는 Amazon EKS에서 Llama-3 모델, [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) 및 [Inferentia2](https://aws.amazon.com/ec2/instance-types/inf2/)를 결합한 솔루션의 아키텍처를 자세히 살펴봅니다.

![Llama-3-inf2](../../img/llama3.png)

## 솔루션 배포

[Amazon EKS](https://aws.amazon.com/eks/)에 `Llama-3-8b-instruct`를 배포하려면 필요한 사전 요구 사항을 다루고 배포 프로세스를 단계별로 안내합니다.

여기에는 인프라 설정, **Ray 클러스터** 배포 및 [Gradio](https://www.gradio.app/) WebUI 앱 생성이 포함됩니다.

<CollapsibleContent header={<h2><span>사전 요구 사항</span></h2>}>
시작하기 전에 배포 프로세스를 원활하고 문제 없이 진행하기 위해 모든 사전 요구 사항이 갖춰져 있는지 확인하십시오.
머신에 다음 도구가 설치되어 있는지 확인하십시오.

1. [aws cli](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
2. [kubectl](https://Kubernetes.io/docs/tasks/tools/)
3. [terraform](https://learn.hashicorp.com/tutorials/terraform/install-cli)

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

kubectl get nodes # 출력은 EKS Managed Node group 노드를 표시합니다
```

</CollapsibleContent>

## Llama3 모델이 있는 Ray 클러스터 배포
`Trainium on EKS` 클러스터가 배포되면 `kubectl`을 사용하여 `ray-service-Llama-3.yaml`을 배포할 수 있습니다.

이 단계에서는 Karpenter 오토스케일링을 사용하는 `x86 CPU` 인스턴스의 `Head Pod` 하나와 [Karpenter](https://karpenter.sh/)에 의해 오토스케일링되는 `Inf2.48xlarge` 인스턴스의 `Ray 워커`로 구성된 Ray Serve 클러스터를 배포합니다.

배포를 진행하기 전에 이 배포에서 사용되는 주요 파일을 자세히 살펴보고 기능을 이해해 봅시다:

- **ray_serve_Llama-3.py:**

이 스크립트는 FastAPI, Ray Serve 및 PyTorch 기반 Hugging Face Transformers를 사용하여 [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) 언어 모델을 사용한 효율적인 텍스트 생성 API를 생성합니다.

스크립트는 입력 문장을 수락하고 향상된 성능을 위한 Neuron 가속의 이점을 활용하여 텍스트 출력을 효율적으로 생성하는 엔드포인트를 설정합니다. 높은 구성 가능성으로 사용자는 챗봇 및 텍스트 생성 작업을 포함한 다양한 자연어 처리 애플리케이션에 맞게 모델 파라미터를 미세 조정할 수 있습니다.

- **ray-service-Llama-3.yaml:**

이 Ray Serve YAML 파일은 `llama-3-8B-Instruct` 모델을 사용한 효율적인 텍스트 생성을 용이하게 하는 Ray Serve 서비스를 배포하기 위한 Kubernetes 구성 역할을 합니다.

리소스를 분리하기 위해 `llama3`라는 Kubernetes 네임스페이스를 정의합니다. 구성 내에서 `llama-3`라는 `RayService` 사양이 생성되고 `llama3` 네임스페이스 내에 호스팅됩니다. `RayService` 사양은 Ray Serve 서비스를 생성하기 위해 Python 스크립트 `ray_serve_llama3.py` (같은 폴더 내의 Dockerfile에 복사됨)를 활용합니다.

이 예제에서 사용된 Docker 이미지는 배포 용이성을 위해 Amazon Elastic Container Registry (ECR)에 공개적으로 제공됩니다.
사용자는 특정 요구 사항에 맞게 Dockerfile을 수정하고 자체 ECR 리포지토리에 푸시하여 YAML 파일에서 참조할 수도 있습니다.

### Llama-3-Instruct 모델 배포

**클러스터가 로컬에서 구성되었는지 확인**
```bash
aws eks --region us-west-2 update-kubeconfig --name trainium-inferentia
```

**RayServe 클러스터 배포**

:::info

llama3-8B-Instruct 모델을 배포하려면 Hugging Face Hub 토큰을 환경 변수로 구성하는 것이 필수적입니다. 이 토큰은 인증 및 모델 액세스에 필요합니다. Hugging Face 토큰 생성 및 관리 방법에 대한 지침은 [Hugging Face Token Management](https://huggingface.co/docs/hub/security-tokens)를 참조하십시오.
:::


```bash
# Hugging Face Hub 토큰을 환경 변수로 설정합니다. 이 변수는 ray-service-llama3.yaml 파일을 적용할 때 대체됩니다

export  HUGGING_FACE_HUB_TOKEN=<Your-Hugging-Face-Hub-Token-Value>

cd ai-on-eks/blueprints/inference/llama3-8b-rayserve-inf2
envsubst < ray-service-llama3.yaml| kubectl apply -f -
```

다음 명령을 실행하여 배포 확인

:::info

배포 프로세스는 최대 10분이 소요될 수 있습니다. Head Pod는 2~3분 내에 준비되고, Ray Serve 워커 Pod는 Huggingface에서 이미지 검색 및 모델 배포에 최대 10분이 소요될 수 있습니다.

:::

```text
$ kubectl get all -n llama3

NAME                                                          READY   STATUS              RESTARTS   AGE
pod/llama3-raycluster-smqrl-head-4wlbb                        0/1     Running             0          77s
pod/service-raycluster-smqrl-worker-inf2-wjxqq                0/1     Running             0          77s

NAME                     TYPE       CLUSTER-IP      EXTERNAL-IP   PORT(S)                                                                                       AGE
service/llama3           ClusterIP   172.20.246.48   <none>       8000:32138/TCP,52365:32653/TCP,8080:32604/TCP,6379:32739/TCP,8265:32288/TCP,10001:32419/TCP   78s

$ kubectl get ingress -n llama3

NAME             CLASS   HOSTS   ADDRESS                                                                         PORTS   AGE
llama3           nginx   *       k8s-ingressn-ingressn-randomid-randomid.elb.us-west-2.amazonaws.com             80      2m4s

```

이제 아래의 로드 밸런서 URL을 사용하여 Ray 대시보드에 액세스할 수 있습니다.

    http://\<NLB_DNS_NAME\>/dashboard/#/serve

공개 로드 밸런서에 액세스할 수 없는 경우 포트 포워딩을 사용하고 다음 명령으로 localhost를 사용하여 Ray 대시보드를 탐색할 수 있습니다:

```bash
kubectl port-forward svc/llama3 8265:8265 -n llama3

# 브라우저에서 링크 열기
http://localhost:8265/

```

이 웹페이지에서 아래 이미지와 같이 모델 배포 진행 상황을 모니터링할 수 있습니다:

![Ray Dashboard](../../img/ray-dashboard.png)

### Llama3 모델 테스트
모델 배포 상태가 `running` 상태가 되면 Llama-3-instruct 사용을 시작할 수 있습니다.

URL 끝에 쿼리를 추가하여 다음 URL을 사용할 수 있습니다.

    http://\<NLB_DNS_NAME\>/serve/infer?sentence=what is data parallelism and tensor parallelisma and the differences

브라우저에서 다음과 같은 출력을 볼 수 있습니다:

![Chat Output](../../img/llama-2-chat-ouput.png)

## Gradio WebUI 앱 배포
배포된 모델과 원활하게 통합되는 사용자 친화적인 채팅 인터페이스를 [Gradio](https://www.gradio.app/)를 사용하여 만드는 방법을 알아봅니다.

RayServe를 사용하여 배포된 LLama-3-Instruct 모델과 상호 작용하기 위해 로컬 머신에 Gradio 앱을 배포해 봅시다.

:::info

Gradio 앱은 데모 목적으로만 생성된 로컬로 노출된 서비스와 상호 작용합니다. 또는 더 넓은 접근성을 위해 Ingress 및 Load Balancer가 있는 Pod로 EKS에 Gradio 앱을 배포할 수 있습니다.

:::

### llama3 Ray 서비스로 포트 포워딩 실행
먼저 kubectl을 사용하여 Llama-3 Ray 서비스로 포트 포워딩을 실행합니다:

```bash
kubectl port-forward svc/llama3-service 8000:8000 -n llama3
```

## Gradio WebUI 앱 배포
배포된 모델과 원활하게 통합되는 사용자 친화적인 채팅 인터페이스를 [Gradio](https://www.gradio.app/)를 사용하여 만드는 방법을 알아봅니다.

localhost에서 Docker 컨테이너로 실행되는 Gradio 앱을 설정하는 것으로 진행합니다. 이 설정을 통해 RayServe를 사용하여 배포된 Llama-3-Instruct 모델과 상호 작용할 수 있습니다.

### Gradio 앱 Docker 컨테이너 빌드

먼저 클라이언트 앱용 Docker 컨테이너를 빌드합니다.

```bash
cd ai-on-eks/blueprints/inference/gradio-ui
docker build --platform=linux/amd64 \
    -t gradio-app:llama \
    --build-arg GRADIO_APP="gradio-app-llama.py" \
    .
```

### Gradio 컨테이너 배포

docker를 사용하여 localhost에서 컨테이너로 Gradio 앱을 배포합니다:

```bash
docker run --rm -it -p 7860:7860 -p 8000:8000 gradio-app:llama
```
:::info
머신에서 Docker Desktop을 실행하지 않고 [finch](https://runfinch.com/)와 같은 것을 대신 사용하는 경우 컨테이너 내부의 사용자 정의 호스트-IP 매핑을 위한 추가 플래그가 필요합니다.

```
docker run --rm -it \
    --add-host ray-service:<workstation-ip> \
    -e "SERVICE_NAME=http://ray-service:8000" \
    -p 7860:7860 gradio-app:llama
```
:::

#### WebUI 호출

웹 브라우저를 열고 다음 URL로 이동하여 Gradio WebUI에 액세스합니다:

로컬 URL에서 실행 중:  http://localhost:7860

이제 로컬 머신에서 Gradio 애플리케이션과 상호 작용할 수 있습니다.

![Gradio Llama-3 AI Chat](../../img/llama3.png)

## 결론

요약하면, Llama-3를 배포하고 확장할 때 AWS Trn1/Inf2 인스턴스는 매력적인 이점을 제공합니다.
GPU 부족과 관련된 문제를 극복하면서 대규모 언어 모델을 효율적이고 접근 가능하게 실행하는 데 필요한 확장성, 비용 최적화 및 성능 향상을 제공합니다. 챗봇, 자연어 처리 애플리케이션 또는 기타 LLM 기반 솔루션을 구축하든 Trn1/Inf2 인스턴스를 통해 AWS 클라우드에서 Llama-3의 잠재력을 최대한 활용할 수 있습니다.

## 정리

마지막으로 더 이상 필요하지 않은 리소스를 정리하고 프로비저닝 해제하는 방법을 안내합니다.

**1단계:** Gradio 컨테이너 삭제

Gradio 앱을 실행하는 컨테이너를 종료하려면 `docker run`이 실행 중인 localhost 터미널 창에서 `Ctrl-c`를 누릅니다. 선택적으로 Docker 이미지를 정리합니다

```bash
docker rmi gradio-app:llama
```
**2단계:** Ray 클러스터 삭제

```bash
cd ai-on-eks/blueprints/inference/llama3-8b-instruct-rayserve-inf2
kubectl delete -f ray-service-llama3.yaml
```

**3단계:** EKS 클러스터 정리
이 스크립트는 `-target` 옵션을 사용하여 모든 리소스가 올바른 순서로 삭제되도록 환경을 정리합니다.

```bash
cd ai-on-eks/infra/trainium-inferentia/
./cleanup.sh
```
