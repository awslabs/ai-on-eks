---
title: Inferentia2의 Stable Diffusion
sidebar_position: 5
---
import CollapsibleContent from '@site/src/components/CollapsibleContent';

:::warning
EKS에 ML 모델을 배포하려면 GPU 또는 Neuron 인스턴스에 대한 액세스가 필요합니다. 배포가 작동하지 않는 경우 이러한 리소스에 대한 액세스가 누락되어 있기 때문인 경우가 많습니다. 또한 일부 배포 패턴은 Karpenter 오토스케일링 및 정적 노드 그룹에 의존합니다. 노드가 초기화되지 않으면 Karpenter 또는 노드 그룹의 로그를 확인하여 문제를 해결하십시오.
:::

:::info

이 예제 블루프린트는 EKS 클러스터에서 워커 노드로 실행되는 Inferentia2 인스턴스에 `stable-diffusion-xl-base-1-0` 모델을 배포합니다. 모델은 `RayServe`를 사용하여 서빙됩니다.

:::

# Inferentia, Ray Serve 및 Gradio를 사용한 Stable Diffusion XL Base 모델 서빙
[Ray Serve](https://docs.ray.io/en/latest/serve/index.html)를 사용하여 Amazon Elastic Kubernetes Service (EKS)에 [Stable Diffusion XL Base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) 모델을 배포하는 포괄적인 가이드에 오신 것을 환영합니다.
이 튜토리얼에서는 Stable Diffusion 모델의 강력한 기능을 활용하는 방법뿐만 아니라 대규모 언어 모델(LLM)을 효율적으로 배포하는 복잡한 과정에 대한 통찰력을 얻을 수 있습니다. 특히 대규모 언어 모델 배포 및 확장에 최적화된 `inf2.24xlarge` 및 `inf2.48xlarge`와 같은 [trn1/inf2](https://aws.amazon.com/machine-learning/neuron/) (AWS Trainium 및 Inferentia 기반) 인스턴스에서의 배포를 다룹니다.

### Stable Diffusion이란?
Stable Diffusion은 몇 초 만에 멋진 아트를 만들 수 있는 텍스트-이미지 모델입니다. 현재 사용 가능한 가장 크고 강력한 LLM 중 하나입니다. 주로 텍스트 설명에 조건화된 상세한 이미지를 생성하는 데 사용되지만 인페인팅, 아웃페인팅 및 텍스트 프롬프트에 의해 안내되는 이미지-이미지 번역 생성과 같은 다른 작업에도 적용할 수 있습니다.

#### Stable Diffusion XL(SDXL)
SDXL은 텍스트-이미지 합성을 위한 잠재 확산 모델입니다. 이전 버전의 Stable Diffusion과 비교하여 SDXL은 잠재 확산 및 노이즈 감소를 위한 파이프라인을 사용합니다. SDXL은 또한 더 많은 어텐션 블록과 SDXL이 두 번째 텍스트 인코더를 사용하기 때문에 더 큰 교차 어텐션 컨텍스트를 갖는 더 큰 UNet을 사용하여 이전 Stable Diffusion 모델에 비해 생성된 이미지의 품질을 향상시킵니다.

SDXL은 여러 새로운 컨디셔닝 체계로 설계되었으며 여러 종횡비로 훈련되었습니다. 또한 이미지-이미지 기술을 사용하여 SDXL에서 생성된 샘플의 시각적 충실도를 개선하는 데 사용되는 정제 모델을 사용합니다.

이 프로세스를 통해 **Amazon EKS**와 **Ray Serve**에서 효과적으로 배포하고 활용할 수 있도록 안내하는 고도로 유능하고 미세 조정된 언어 모델이 생성됩니다.

## Trn1/Inf2 인스턴스에서의 추론: Stable Diffusion LLM의 잠재력 극대화
**Stable Diffusion XL**은 다양한 하드웨어 플랫폼에 배포할 수 있으며, 각각 고유한 장점이 있습니다. 그러나 Stable Diffusion 모델의 효율성, 확장성 및 비용 효율성을 최대화하는 데 있어 [AWS Trn1/Inf2 인스턴스](https://aws.amazon.com/ec2/instance-types/inf2/)가 최적의 선택입니다.

**확장성 및 가용성**
StableDiffusion XL과 같은 대규모 언어 모델(`LLM`)을 배포할 때 주요 과제 중 하나는 적절한 하드웨어의 확장성과 가용성입니다. 기존 `GPU` 인스턴스는 높은 수요로 인해 부족한 경우가 많아 리소스를 효과적으로 프로비저닝하고 확장하기가 어렵습니다.
반면 `trn1.32xlarge`, `trn1n.32xlarge`, `inf2.24xlarge` 및 `inf2.48xlarge`와 같은 `Trn1/Inf2` 인스턴스는 LLM을 포함한 생성형 AI 모델의 고성능 딥러닝(DL) 훈련 및 추론을 위해 특별히 구축되었습니다. 확장성과 가용성을 모두 제공하여 리소스 병목 현상이나 지연 없이 필요에 따라 `Stable-diffusion-xl` 모델을 배포하고 확장할 수 있습니다.

**비용 최적화:**
기존 GPU 인스턴스에서 LLM을 실행하면 GPU의 부족과 경쟁적인 가격으로 인해 비용이 많이 들 수 있습니다.
**Trn1/Inf2** 인스턴스는 비용 효율적인 대안을 제공합니다. AI 및 기계 학습 작업에 최적화된 전용 하드웨어를 제공함으로써 Trn1/Inf2 인스턴스를 통해 비용의 일부로 최고 수준의 성능을 달성할 수 있습니다.
이러한 비용 최적화를 통해 예산을 효율적으로 할당하여 LLM 배포를 접근 가능하고 지속 가능하게 만들 수 있습니다.

**성능 향상**
Stable-Diffusion-xl은 GPU에서 고성능 추론을 달성할 수 있지만, Neuron 가속기는 성능을 한 단계 더 끌어올립니다. Neuron 가속기는 기계 학습 워크로드를 위해 특별히 구축되어 Stable-diffusion의 추론 속도를 크게 향상시키는 하드웨어 가속을 제공합니다. 이는 Trn1/Inf2 인스턴스에 Stable-Diffusion-xl을 배포할 때 더 빠른 응답 시간과 개선된 사용자 경험으로 이어집니다.

### 예제 사용 사례
디지털 아트 회사가 프롬프트를 기반으로 가능한 아트를 생성하는 데 도움이 되는 Stable-diffusion-xl 기반 이미지 생성기를 배포하려고 합니다. 텍스트 프롬프트 선택을 사용하여 사용자는 다양한 스타일의 아트워크, 그래픽 및 로고를 만들 수 있습니다. 이미지 생성기를 사용하여 아트를 예측하거나 미세 조정할 수 있으며 제품 반복 주기에서 상당한 시간 절약을 가져올 수 있습니다. 회사는 대규모 고객 기반을 보유하고 있으며 모델이 높은 부하에서 확장 가능하기를 원합니다. 회사는 높은 요청량을 처리하고 빠른 응답 시간을 제공할 수 있는 인프라를 설계해야 합니다.

회사는 Inferentia2 인스턴스를 사용하여 Stable diffusion 이미지 생성기를 효율적으로 확장할 수 있습니다. Inferentia2 인스턴스는 기계 학습 작업을 위한 특수 하드웨어 가속기입니다. 기계 학습 워크로드에 대해 GPU보다 최대 20배 더 나은 성능과 최대 7배 더 낮은 비용을 제공할 수 있습니다.

회사는 또한 Ray Serve를 사용하여 Stable diffusion 이미지 생성기를 수평으로 확장할 수 있습니다. Ray Serve는 기계 학습 모델을 서빙하기 위한 분산 프레임워크입니다. 수요에 따라 모델을 자동으로 확장하거나 축소할 수 있습니다.

Stable diffusion 이미지 생성기를 확장하기 위해 회사는 여러 Inferentia2 인스턴스를 배포하고 Ray Serve를 사용하여 인스턴스 간에 트래픽을 분산할 수 있습니다. 이를 통해 회사는 높은 요청량을 처리하고 빠른 응답 시간을 제공할 수 있습니다.

## 솔루션 아키텍처
이 섹션에서는 Amazon EKS에서 Stable diffusion xl 모델, [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) 및 [Inferentia2](https://aws.amazon.com/ec2/instance-types/inf2/)를 결합한 솔루션의 아키텍처를 자세히 살펴봅니다.

![Sdxl-inf2](../../img/excali-draw-sdxl-inf2.png)

## 솔루션 배포
[Amazon EKS](https://aws.amazon.com/eks/)에 `stable-diffusion-xl-base-1-0`를 배포하려면 필요한 사전 요구 사항을 다루고 배포 프로세스를 단계별로 안내합니다.
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

## Stable Diffusion XL 모델이 있는 Ray 클러스터 배포

`Trainium on EKS` 클러스터가 배포되면 `kubectl`을 사용하여 `ray-service-stablediffusion.yaml`을 배포할 수 있습니다.

이 단계에서는 Karpenter 오토스케일링을 사용하는 `x86 CPU` 인스턴스의 `Head Pod` 하나와 [Karpenter](https://karpenter.sh/)에 의해 오토스케일링되는 `Inf2.48xlarge` 인스턴스의 `Ray 워커`로 구성된 Ray Serve 클러스터를 배포합니다.

배포를 진행하기 전에 이 배포에서 사용되는 주요 파일을 자세히 살펴보고 기능을 이해해 봅시다:

- **ray_serve_stablediffusion.py:**
이 스크립트는 FastAPI, Ray Serve 및 [Hugging Face Optimum Neuron](https://github.com/huggingface/optimum-neuron) 도구 라이브러리를 사용하여 [Neuronx model for stable-diffusion-xl-base-1-0-1024x1024](https://huggingface.co/aws-neuron/stable-diffusion-xl-base-1-0-1024x1024) 언어 모델을 사용한 효율적인 텍스트-이미지 생성기를 생성합니다.

이 예제 블루프린트에서는 AWS Neuron에서 실행되도록 컴파일된 사전 컴파일 모델을 사용합니다. 원하는 stable diffusion 모델을 사용하고 추론을 수행하기 전에 AWS Neuron에서 실행되도록 컴파일할 수 있습니다.

- **ray-service-stablediffusion.yaml:**
이 Ray Serve YAML 파일은 `stable-diffusion-xl-base-1.0` 모델을 사용한 효율적인 텍스트 생성을 용이하게 하는 Ray Serve 서비스를 배포하기 위한 Kubernetes 구성 역할을 합니다.
리소스를 분리하기 위해 `stablediffusion`이라는 Kubernetes 네임스페이스를 정의합니다. 구성 내에서 `stablediffusion-service`라는 `RayService` 사양이 생성되고 `stablediffusion` 네임스페이스 내에 호스팅됩니다. `RayService` 사양은 Ray Serve 서비스를 생성하기 위해 Python 스크립트 `ray_serve_stablediffusion.py` (같은 폴더 내의 Dockerfile에 복사됨)를 활용합니다.
이 예제에서 사용된 Docker 이미지는 배포 용이성을 위해 Amazon Elastic Container Registry (ECR)에 공개적으로 제공됩니다.
사용자는 특정 요구 사항에 맞게 Dockerfile을 수정하고 자체 ECR 리포지토리에 푸시하여 YAML 파일에서 참조할 수도 있습니다.

### Stable-Diffusion-xl-base-1-0 모델 배포

**클러스터가 로컬에서 구성되었는지 확인**
```bash
aws eks --region us-west-2 update-kubeconfig --name trainium-inferentia
```

**RayServe 클러스터 배포**

```bash
cd ai-on-eks/blueprints/inference/stable-diffusion-xl-base-rayserve-inf2
kubectl apply -f ray-service-stablediffusion.yaml
```

다음 명령을 실행하여 배포 확인

:::info

배포 프로세스는 최대 10분이 소요될 수 있습니다. Head Pod는 2~3분 내에 준비되고, Ray Serve 워커 Pod는 Huggingface에서 이미지 검색 및 모델 배포에 최대 10분이 소요될 수 있습니다.

:::

```text
$ kubectl get po -n stablediffusion -w

NAME                                                      READY   STATUS     RESTARTS   AGE
service-raycluster-gc7gb-worker-inf2-worker-group-k2kf2   0/1     Init:0/1   0          7s
stablediffusion-service-raycluster-gc7gb-head-6fqvv       1/1     Running    0          7s

service-raycluster-gc7gb-worker-inf2-worker-group-k2kf2   0/1     PodInitializing   0          9s
service-raycluster-gc7gb-worker-inf2-worker-group-k2kf2   1/1     Running           0          10s
stablediffusion-service-raycluster-gc7gb-head-6fqvv       1/1     Running           0          53s
service-raycluster-gc7gb-worker-inf2-worker-group-k2kf2   1/1     Running           0          53s
```

생성된 서비스 및 ingress 리소스도 확인

```text
kubectl get svc -n stablediffusion

NAME                                TYPE       CLUSTER-IP       EXTERNAL-IP   PORT(S)                                                                                       AGE
stablediffusion-service             NodePort   172.20.175.61    <none>        6379:32190/TCP,8265:32375/TCP,10001:32117/TCP,8000:30770/TCP,52365:30334/TCP,8080:30094/TCP   16h
stablediffusion-service-head-svc    NodePort   172.20.193.225   <none>        6379:32228/TCP,8265:30215/TCP,10001:30767/TCP,8000:31482/TCP,52365:30170/TCP,8080:31584/TCP   16h
stablediffusion-service-serve-svc   NodePort   172.20.15.224    <none>        8000:30982/TCP                                                                                16h


$ kubectl get ingress -n stablediffusion

NAME                      CLASS   HOSTS   ADDRESS                                                                         PORTS   AGE
stablediffusion-ingress   nginx   *       k8s-ingressn-ingressn-7f3f4b475b-1b8966c0b8f4d3da.elb.us-west-2.amazonaws.com   80      16h
```

이제 아래의 로드 밸런서 URL을 사용하여 Ray 대시보드에 액세스할 수 있습니다.

    http://\<NLB_DNS_NAME\>/dashboard/#/serve

공개 로드 밸런서에 액세스할 수 없는 경우 포트 포워딩을 사용하고 다음 명령으로 localhost를 사용하여 Ray 대시보드를 탐색할 수 있습니다:

```bash
kubectl port-forward svc/stablediffusion-service 8265:8265 -n stablediffusion

# 브라우저에서 링크 열기
http://localhost:8265/

```

이 웹페이지에서 아래 이미지와 같이 모델 배포 진행 상황을 모니터링할 수 있습니다:

![Ray Dashboard](../../img/ray-dashboard-sdxl.png)

### Stable Diffusion XL 모델 테스트

Ray 대시보드에서 Stable Diffusion 모델 배포 상태가 `running` 상태로 전환되었는지 확인하면 모델을 활용할 준비가 된 것입니다. 이 상태 변경은 Stable Diffusion 모델이 이제 완전히 작동하며 텍스트 설명을 기반으로 이미지 생성 요청을 처리할 준비가 되었음을 의미합니다.

URL 끝에 쿼리를 추가하여 다음 URL을 사용할 수 있습니다.

    http://\<NLB_DNS_NAME\>/serve/imagine?prompt=an astronaut is dancing on green grass, sunlit

브라우저에서 다음과 같은 출력을 볼 수 있습니다:

![Prompt Output](../../img/stable-diffusion-xl-prompt_3.png)

## Gradio WebUI 앱 배포
배포된 모델과 원활하게 통합되는 사용자 친화적인 채팅 인터페이스를 [Gradio](https://www.gradio.app/)를 사용하여 만드는 방법을 알아봅니다.

localhost에서 Docker 컨테이너로 실행되는 Gradio 앱을 설정하는 것으로 진행합니다. 이 설정을 통해 RayServe를 사용하여 배포된 Stable Diffusion XL 모델과 상호 작용할 수 있습니다.

### Gradio 앱 Docker 컨테이너 빌드

먼저 클라이언트 앱용 Docker 컨테이너를 빌드합니다.

```bash
cd ai-on-eks/blueprints/inference/gradio-ui
docker build --platform=linux/amd64 \
    -t gradio-app:sd \
    --build-arg GRADIO_APP="gradio-app-stable-diffusion.py" \
    .
```

### Gradio 컨테이너 배포

docker를 사용하여 localhost에서 컨테이너로 Gradio 앱을 배포합니다:

```bash
docker run --rm -it -p 7860:7860 -p 8000:8000 gradio-app:sd
```

:::info
머신에서 Docker Desktop을 실행하지 않고 [finch](https://runfinch.com/)와 같은 것을 대신 사용하는 경우 컨테이너 내부의 사용자 정의 호스트-IP 매핑을 위한 추가 플래그가 필요합니다.

```
docker run --rm -it \
    --add-host ray-service:<workstation-ip> \
    -e "SERVICE_NAME=http://ray-service:8000" \
    -p 7860:7860 gradio-app:sd
```
:::


#### WebUI 호출

웹 브라우저를 열고 다음 URL로 이동하여 Gradio WebUI에 액세스합니다:

로컬 URL에서 실행 중:  http://localhost:7860

이제 로컬 머신에서 Gradio 애플리케이션과 상호 작용할 수 있습니다.

![Gradio Output](../../img/stable-diffusion-xl-gradio.png)

## 결론

결론적으로, **Stable-diffusion-xl-base** 모델을 Ray Serve와 함께 EKS에 성공적으로 배포하고 Gradio를 사용하여 프롬프트 기반 웹 UI를 생성했습니다.
이는 자연어 처리 및 프롬프트 기반 이미지 생성기 및 이미지 예측기 개발에 흥미로운 가능성을 열어줍니다.

요약하면, Stable diffusion 모델을 배포하고 확장할 때 AWS Trn1/Inf2 인스턴스는 매력적인 이점을 제공합니다.
GPU 부족과 관련된 문제를 극복하면서 대규모 언어 모델을 효율적이고 접근 가능하게 실행하는 데 필요한 확장성, 비용 최적화 및 성능 향상을 제공합니다.
텍스트-이미지 생성기, 이미지-이미지 생성기 또는 기타 LLM 기반 솔루션을 구축하든 Trn1/Inf2 인스턴스를 통해 AWS 클라우드에서 Stable Diffusion LLM의 잠재력을 최대한 활용할 수 있습니다.

## 정리
마지막으로 더 이상 필요하지 않은 리소스를 정리하고 프로비저닝 해제하는 방법을 안내합니다.

**1단계:** Gradio 컨테이너 삭제

Gradio 앱을 실행하는 컨테이너를 종료하려면 `docker run`이 실행 중인 localhost 터미널 창에서 `Ctrl-c`를 누릅니다. 선택적으로 Docker 이미지를 정리합니다

```bash
docker rmi gradio-app:sd
```
**2단계:** Ray 클러스터 삭제

```bash
cd ai-on-eks/blueprints/inference/stable-diffusion-xl-base-rayserve-inf2
kubectl delete -f ray-service-stablediffusion.yaml
```

**3단계:** EKS 클러스터 정리
이 스크립트는 `-target` 옵션을 사용하여 모든 리소스가 올바른 순서로 삭제되도록 환경을 정리합니다.

```bash
cd ai-on-eks/infra/trainium-inferentia/
./cleanup.sh
```
