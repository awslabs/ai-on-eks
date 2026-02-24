---
title: GPU에서의 Stable Diffusion
sidebar_position: 3
---
import CollapsibleContent from '@site/src/components/CollapsibleContent';

:::warning
EKS에서 ML 모델을 배포하려면 GPU 또는 Neuron 인스턴스에 대한 접근이 필요합니다. 배포가 작동하지 않는 경우, 이러한 리소스에 대한 접근 권한이 없기 때문인 경우가 많습니다. 또한 일부 배포 패턴은 Karpenter 자동 스케일링과 정적 노드 그룹에 의존합니다. 노드가 초기화되지 않으면 Karpenter 또는 노드 그룹의 로그를 확인하여 문제를 해결하세요.
:::

:::info

관측성과 로깅 개선 사항을 통합하기 위해 이 블루프린트를 적극적으로 개선하고 있습니다.

:::


# GPU, Ray Serve 및 Gradio를 사용한 Stable Diffusion v2 배포
이 패턴은 [GPU](https://aws.amazon.com/ec2/instance-types/g4/)를 사용하여 Amazon EKS에서 [Stable Diffusion V2](https://huggingface.co/stabilityai/stable-diffusion-2-1) 모델을 배포하여 가속화된 이미지 생성을 시연합니다. [Ray Serve](https://docs.ray.io/en/latest/serve/index.html)는 여러 GPU 노드에서 효율적인 스케일링을 제공하고, [Karpenter](https://karpenter.sh/)는 노드 프로비저닝을 동적으로 관리합니다.

이 패턴을 통해 다음을 달성합니다:

- 노드의 동적 스케일링을 위한 Karpenter 관리 GPU nodepool이 있는 Amazon EKS 클러스터 생성.
- [jark-stack](https://github.com/awslabs/ai-on-eks/tree/main/infra/jark-stack/terraform) Terraform 블루프린트를 사용하여 KubeRay Operator 및 기타 핵심 EKS 애드온 설치.
- GPU 리소스 전체에서 효율적인 스케일링을 위해 RayServe를 사용하여 Stable Diffusion 모델 배포

### Stable Diffusion이란?
Stable Diffusion은 텍스트 설명에서 멋지고 상세한 이미지를 생성하는 최첨단 텍스트-이미지 모델입니다. 이미지 생성을 통해 상상력을 발휘하고자 하는 아티스트, 디자이너 및 모든 사람을 위한 강력한 도구입니다. 이 모델은 이미지 생성 프로세스에서 높은 수준의 창의적 제어와 유연성을 제공합니다.

## 솔루션 배포
Amazon EKS에서 Stable Diffusion v2-1을 시작해 보겠습니다! 이 섹션에서 다룰 내용은:

- **사전 요구 사항**: 모든 것이 준비되었는지 확인.
- **인프라 설정**: EKS 클러스터를 생성하고 배포 준비.
- **Ray 클러스터 배포**: 확장성과 효율성을 제공하는 이미지 생성 파이프라인의 핵심.
- **Gradio Web UI 구축**: Stable Diffusion과 상호 작용하기 위한 사용자 친화적인 인터페이스.

<CollapsibleContent header={<h2><span>사전 요구 사항</span></h2>}>
시작하기 전에 배포 과정을 원활하게 진행하기 위해 모든 사전 요구 사항이 갖춰져 있는지 확인하세요.
머신에 다음 도구가 설치되어 있는지 확인하세요.

1. [aws cli](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
2. [kubectl](https://Kubernetes.io/docs/tasks/tools/)
3. [terraform](https://learn.hashicorp.com/tutorials/terraform/install-cli)

<a id="선택-사항-bottlerocket-os에-컨테이너-이미지를-미리-로드하여-콜드-스타트-시간-단축"></a>
### (선택 사항) Bottlerocket OS에 컨테이너 이미지를 미리 로드하여 콜드 스타트 시간 단축

Ray 워커에서 이미지 검색 배포를 가속화하려면 [Karpenter와 EBS 스냅샷을 사용하여 Bottlerocket 데이터 볼륨에 컨테이너 이미지 미리 로드](../../bestpractices/preload-container-images)를 참조하세요.

`TF_VAR_bottlerocket_data_disk_snpashot_id`를 정의하여 Karpenter가 EBS 스냅샷이 포함된 Bottlerocket 워커 노드를 프로비저닝하도록 하면 컨테이너 시작을 위한 콜드 스타트를 줄일 수 있습니다. 이렇게 하면 Amazon ECR에서 컨테이너 이미지를 다운로드하고 추출하는 데 10분(이미지 크기에 따라 다름)을 절약할 수 있습니다.

```
export TF_VAR_bottlerocket_data_disk_snpashot_id=snap-0c6d965cf431785ed
```
<a id="배포"></a>
### 배포

저장소 복제

```bash
git clone https://github.com/awslabs/ai-on-eks.git
```


```
cd ai-on-eks/infra/jark-stack/ && chmod +x install.sh
./install.sh
```

예제 디렉토리 중 하나로 이동하여 `install.sh` 스크립트를 실행합니다

**중요 참고 사항:** 블루프린트를 배포하기 전에 `blueprint.tfvars` 파일에서 리전을 업데이트하세요.
또한 불일치를 방지하기 위해 로컬 리전 설정이 지정된 리전과 일치하는지 확인하세요.
예를 들어, `export AWS_DEFAULT_REGION="<REGION>"`을 원하는 리전으로 설정하세요:

```bash
cd ai-on-eks/infra/jark-stack/ && chmod +x install.sh
./install.sh
```

<a id="리소스-확인"></a>
### 리소스 확인

Amazon EKS 클러스터 확인

```bash
aws eks --region us-west-2 describe-cluster --name jark-stack
```

```bash
# EKS로 인증하기 위한 k8s 설정 파일 생성
aws eks --region us-west-2 update-kubeconfig --name jark-stack

# EKS Managed Node 그룹 노드 출력
kubectl get nodes
```

</CollapsibleContent>

## Stable Diffusion 모델을 포함한 Ray 클러스터 배포

`jark-stack` 클러스터가 배포되면 `kubectl`을 사용하여 `/ai-on-eks/blueprints/inference/stable-diffusion-rayserve-gpu/` 경로에서 `ray-service-stablediffusion.yaml`을 배포할 수 있습니다.

이 단계에서는 Karpenter 자동 스케일링을 사용하여 `x86 CPU` 인스턴스에서 하나의 `Head Pod`와 [Karpenter](https://karpenter.sh/)에 의해 자동 스케일링되는 `g5.2xlarge` 인스턴스에서 `Ray workers`로 구성된 Ray Serve 클러스터를 배포합니다.

이 배포에 사용되는 주요 파일을 자세히 살펴보고 배포를 진행하기 전에 기능을 이해하겠습니다:
- **ray_serve_sd.py:**
  이 스크립트는 GPU 장착 인프라에서 확장 가능한 모델 서빙을 가능하게 하는 Ray Serve를 사용하여 배포된 두 가지 주요 구성 요소가 있는 FastAPI 애플리케이션을 설정합니다:
  - **StableDiffusionV2 Deployment**: 이 클래스는 스케줄러를 사용하여 Stable Diffusion V2 모델을 초기화하고 처리를 위해 GPU로 이동합니다. 텍스트 프롬프트를 기반으로 이미지를 생성하는 기능이 포함되어 있으며, 이미지 크기는 입력 파라미터를 통해 사용자 정의할 수 있습니다.
  - **APIIngress**: 이 FastAPI 엔드포인트는 Stable Diffusion 모델에 대한 인터페이스 역할을 합니다. 텍스트 프롬프트와 선택적 이미지 크기를 받는 `/imagine` 경로에서 GET 메서드를 노출합니다. Stable Diffusion 모델을 사용하여 이미지를 생성하고 PNG 파일로 반환합니다.

- **ray-service-stablediffusion.yaml:**
  이 RayServe 배포 패턴은 GPU 지원이 포함된 Amazon EKS에서 Stable Diffusion 모델을 호스팅하기 위한 확장 가능한 서비스를 설정합니다. 전용 네임스페이스를 만들고 수신 트래픽에 따라 리소스 활용을 효율적으로 관리하는 자동 스케일링 기능이 있는 RayService를 구성합니다. 배포는 RayService 우산 아래에서 서빙되는 모델이 수요에 따라 1개에서 4개의 레플리카 사이에서 자동으로 조정할 수 있도록 보장하며, 각 레플리카에는 GPU가 필요합니다. 이 패턴은 성능을 극대화하고 무거운 의존성이 미리 로드되도록 하여 시작 지연을 최소화하도록 설계된 사용자 정의 컨테이너 이미지를 사용합니다.

### Stable Diffusion V2 모델 배포

클러스터가 로컬에서 구성되었는지 확인

```bash
aws eks --region us-west-2 update-kubeconfig --name jark-stack
```

**RayServe 클러스터 배포**

```bash
cd ai-on-eks/blueprints/inference/stable-diffusion-rayserve-gpu
kubectl apply -f ray-service-stablediffusion.yaml
```

다음 명령을 실행하여 배포를 확인합니다

:::info

데이터 볼륨에 컨테이너 이미지를 미리 로드하지 않은 경우 배포 프로세스에 최대 10~12분이 걸릴 수 있습니다. Head Pod는 2~3분 내에 준비될 것으로 예상되며, Ray Serve 워커 파드는 Huggingface에서 이미지 검색 및 모델 배포에 최대 10분이 걸릴 수 있습니다.

:::

이 배포는 아래와 같이 x86 인스턴스에서 실행되는 Ray head 파드와 GPU G5 인스턴스에서 실행되는 워커 파드를 설정합니다.

```bash
kubectl get pods -n stablediffusion

NAME                                                      READY   STATUS
rservice-raycluster-hb4l4-worker-gpu-worker-group-z8gdw   1/1     Running
stablediffusion-service-raycluster-hb4l4-head-4kfzz       2/2     Running
```

데이터 볼륨에 컨테이너 이미지를 미리 로드한 경우 `kubectl describe pod -n stablediffusion` 출력에서 `Container image "public.ecr.aws/data-on-eks/ray2.11.0-py310-gpu-stablediffusion:latest" already present on machine` 메시지를 찾을 수 있습니다.


```
kubectl describe pod -n stablediffusion

...
Events:
  Type     Reason            Age                From               Message
  ----     ------            ----               ----               -------
  Warning  FailedScheduling  41m                default-scheduler  0/8 nodes are available: 1 Insufficient cpu, 3 Insufficient memory, 8 Insufficient nvidia.com/gpu. preemption: 0/8 nodes are available: 8 No preemption victims found for incoming pod.
  Normal   Nominated         41m                karpenter          Pod should schedule on: nodeclaim/gpu-ljvhl
  Normal   Scheduled         40m                default-scheduler  Successfully assigned stablediffusion/stablediffusion-raycluster-ms6pl-worker-gpu-85d22 to ip-100-64-136-72.us-west-2.compute.internal
  Normal   Pulled            40m                kubelet            Container image "public.ecr.aws/data-on-eks/ray2.11.0-py310-gpu-stablediffusion:latest" already present on machine
  Normal   Created           40m                kubelet            Created container wait-gcs-ready
  Normal   Started           40m                kubelet            Started container wait-gcs-ready
  Normal   Pulled            39m                kubelet            Container image "public.ecr.aws/data-on-eks/ray2.11.0-py310-gpu-stablediffusion:latest" already present on machine
  Normal   Created           39m                kubelet            Created container worker
  Normal   Started           38m                kubelet            Started container worker
  ```

이 배포는 또한 여러 포트가 구성된 stablediffusion 서비스를 설정합니다. 포트 `8265`는 Ray 대시보드용이고 포트 `8000`은 Stable Diffusion 모델 엔드포인트용입니다.

```bash
kubectl get svc -n stablediffusion
NAME                                TYPE       CLUSTER-IP       EXTERNAL-IP   PORT(S)
stablediffusion-service             NodePort   172.20.223.142   <none>        8080:30213/TCP,6379:30386/TCP,8265:30857/TCP,10001:30666/TCP,8000:31194/TCP
stablediffusion-service-head-svc    NodePort   172.20.215.100   <none>        8265:30170/TCP,10001:31246/TCP,8000:30376/TCP,8080:32646/TCP,6379:31846/TCP
stablediffusion-service-serve-svc   NodePort   172.20.153.125   <none>        8000:31459/TCP
```

Ray 대시보드의 경우 이러한 포트를 개별적으로 포트 포워딩하여 localhost를 사용하여 웹 UI에 로컬로 접근할 수 있습니다.

```bash
kubectl port-forward svc/stablediffusion-service 8266:8265 -n stablediffusion
```

`http://localhost:8265`에서 웹 UI에 접근하세요. 이 인터페이스는 Ray 에코시스템 내의 작업 및 액터 배포를 표시합니다.

![RayServe Deployment](../../img/ray-serve-gpu-sd.png)

제공된 스크린샷은 Serve 배포와 Ray 클러스터 배포를 보여주며, 설정 및 운영 상태에 대한 시각적 개요를 제공합니다.

![RayServe Cluster](../../img/ray-serve-gpu-sd-cluster.png)

## Gradio WebUI 앱 배포
배포된 모델과 원활하게 통합되는 [Gradio](https://www.gradio.app/)를 사용하여 사용자 친화적인 채팅 인터페이스를 만드는 방법을 알아보세요.

localhost에서 Docker 컨테이너로 Gradio 앱을 설정해 보겠습니다. 이 설정을 통해 RayServe를 사용하여 배포된 Stable Diffusion XL 모델과 상호 작용할 수 있습니다.

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

Docker를 사용하여 localhost에서 컨테이너로 Gradio 앱을 배포합니다:

```bash
docker run --rm -it -p 7860:7860 -p 8000:8000 gradio-app:sd
```

:::info
Docker Desktop을 사용하지 않고 [finch](https://runfinch.com/)와 같은 것을 사용하는 경우 컨테이너 내부의 사용자 정의 호스트-IP 매핑을 위한 추가 플래그가 필요합니다.

```
docker run --rm -it \
    --add-host ray-service:<workstation-ip> \
    -e "SERVICE_NAME=http://ray-service:8000" \
    -p 7860:7860 gradio-app:sd
```
:::

#### WebUI 호출

웹 브라우저를 열고 다음 URL로 이동하여 Gradio WebUI에 접근합니다:

로컬 URL에서 실행 중: http://localhost:7860

이제 로컬 머신에서 Gradio 애플리케이션과 상호 작용할 수 있습니다.

![Gradio Output](../../img/gradio-app-gpu.png)

### Ray 자동 스케일링
`ray-serve-stablediffusion.yaml` 파일에 자세히 설명된 Ray 자동 스케일링 구성은 Kubernetes에서 Ray의 기능을 활용하여 계산 요구에 따라 애플리케이션을 동적으로 스케일링합니다.

1. **수신 트래픽**: stable-diffusion 배포에 대한 수신 요청은 Ray Serve가 기존 레플리카의 부하를 모니터링하도록 트리거합니다.
2. **메트릭 기반 스케일링**: Ray Serve는 레플리카당 평균 진행 중인 요청 수를 추적합니다. 이 구성에서는 `target_num_ongoing_requests_per_replica`가 1로 설정되어 있습니다. 이 메트릭이 임계값을 초과하면 더 많은 레플리카가 필요하다는 신호를 보냅니다.
3. **레플리카 생성 (노드 내)**: 노드에 충분한 GPU 용량이 있으면 Ray Serve는 기존 노드 내에 새 레플리카를 추가하려고 시도합니다. 배포는 레플리카당 1개의 GPU를 요청합니다 (`ray_actor_options: num_gpus: 1`).
4. **노드 스케일링 (Karpenter)**: 노드가 추가 레플리카를 수용할 수 없는 경우 (예: 노드당 하나의 GPU만 있는 경우), Ray는 Kubernetes에 더 많은 리소스가 필요하다고 신호를 보냅니다. Karpenter는 Kubernetes의 보류 중인 파드 요청을 관찰하고 리소스 요구를 충족하기 위해 새 g5 GPU 노드를 프로비저닝합니다.
5. **레플리카 생성 (노드 간)**: 새 노드가 준비되면 Ray Serve는 새로 프로비저닝된 노드에 추가 레플리카를 스케줄링합니다.

**자동 스케일링 시뮬레이션:**
1. **부하 생성**: 스크립트를 만들거나 부하 테스트 도구를 사용하여 stable diffusion 서비스에 버스트 이미지 생성 요청을 보냅니다.
2. **관찰 (Ray Dashboard)**: 포트 포워딩 또는 퍼블릭 NLB(구성된 경우)를 통해 http://your-cluster/dashboard에서 Ray Dashboard에 접근합니다. 다음 메트릭이 어떻게 변경되는지 관찰합니다:
        배포의 레플리카 수.
        Ray 클러스터의 노드 수.
3. **관찰 (Kubernetes)**: `kubectl get pods -n stablediffusion`을 사용하여 새 파드 생성을 확인합니다. `kubectl get nodes`를 사용하여 Karpenter가 프로비저닝한 새 노드를 관찰합니다.

## 정리
마지막으로 리소스가 더 이상 필요하지 않을 때 정리하고 프로비저닝을 해제하는 방법을 안내합니다.

**1단계:** Gradio 컨테이너 삭제

`docker run`이 실행 중인 localhost 터미널 창에서 `Ctrl-c`를 눌러 Gradio 앱을 실행하는 컨테이너를 종료합니다. 선택적으로 Docker 이미지를 정리합니다

```bash
docker rmi gradio-app:sd
```
**2단계:** Ray 클러스터 삭제

```bash
cd ai-on-eks/blueprints/inference/stable-diffusion-rayserve-gpu
kubectl delete -f ray-service-stablediffusion.yaml
```

**3단계:** EKS 클러스터 정리
이 스크립트는 `-target` 옵션을 사용하여 모든 리소스가 올바른 순서로 삭제되도록 환경을 정리합니다.

```bash
cd ai-on-eks/infra/jark-stack/
./cleanup.sh
```
