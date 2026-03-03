---
sidebar_label: EKS 기반 AIBrix
---
import CollapsibleContent from '@site/src/components/CollapsibleContent';

# EKS 기반 AIBrix

:::warning
EKS에서 ML 모델을 배포하려면 GPU 또는 Neuron 인스턴스에 대한 액세스가 필요합니다. 배포가 작동하지 않는 경우 이러한 리소스에 대한 액세스가 누락되어 있는 경우가 많습니다. 또한 일부 배포 패턴은 Karpenter 자동 확장 및 정적 노드 그룹에 의존합니다. 노드가 초기화되지 않는 경우 Karpenter 또는 노드 그룹의 로그를 확인하여 문제를 해결하세요.
:::

:::info
이 지침은 AIBrix 클러스터를 기본으로만 배포합니다. 추론 또는 훈련을 위한 특정 모델을 배포하려면 [AI](https://awslabs.github.io/ai-on-eks/docs/blueprints) 페이지에서 엔드투엔드 지침을 참조하세요.
:::

### AIBrix란?
AIBrix는 확장 가능한 GenAI 추론 인프라를 구축하기 위한 필수 빌딩 블록을 제공하도록 설계된 오픈소스 이니셔티브입니다. AIBrix는 특히 엔터프라이즈 요구 사항에 맞춤화된 대규모 언어 모델(LLM) 추론을 배포, 관리 및 확장하는 데 최적화된 클라우드 네이티브 솔루션을 제공합니다.
![Alt text](https://aibrix.readthedocs.io/latest/_images/aibrix-architecture-v1.jpeg)

### 주요 기능 및 이점
* LLM 게이트웨이 및 라우팅: 여러 모델과 복제본에 걸쳐 트래픽을 효율적으로 관리하고 지시합니다.
* 고밀도 LoRA 관리: 모델의 경량, 저랭크 적응에 대한 간소화된 지원.
* 분산 추론: 여러 노드에 걸쳐 대규모 워크로드를 처리하는 확장 가능한 아키텍처.
* LLM 앱 맞춤형 오토스케일러: 실시간 수요에 따라 추론 리소스를 동적으로 확장합니다.
* 통합 AI 런타임: 메트릭 표준화, 모델 다운로드 및 관리를 가능하게 하는 다목적 사이드카.
* 이기종 GPU 추론: 이기종 GPU를 사용한 비용 효율적인 SLO 기반 LLM 추론.
* GPU 하드웨어 장애 감지: GPU 하드웨어 문제의 사전 감지.


<CollapsibleContent header={<h2><span>솔루션 배포</span></h2>}>

이 [예제](https://github.com/awslabs/ai-on-eks/tree/main/infra/aibrix/terraform)에서는 Amazon EKS에 AIBrix를 프로비저닝합니다.

<a id="사전-요구-사항"></a>
### 사전 요구 사항

머신에 다음 도구가 설치되어 있는지 확인하세요.

1. [aws cli](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
2. [kubectl](https://Kubernetes.io/docs/tasks/tools/)
3. [terraform](https://learn.hashicorp.com/tutorials/terraform/install-cli)

<a id="배포"></a>
### 배포

**1. 리포지토리 복제:**

```bash
git clone https://github.com/awslabs/ai-on-eks.git
```

:::info
인증에 프로필을 사용하는 경우
`export AWS_PROFILE="<PROFILE_name>"`을 원하는 프로필 이름으로 설정하세요
:::

**2. 구성 검토 및 사용자 지정:**

- `infra/base/terraform/variables.tf`에서 사용 가능한 애드온을 확인하세요
- 필요에 따라 `infra/aibrix/terraform/blueprint.tfvars`에서 애드온 설정을 수정하세요
- `blueprint.tfvars`에서 AWS 리전을 업데이트하세요

aibrix로 이동하여 `install.sh` 스크립트를 실행합니다

```bash
cd ai-on-eks/infra/aibrix
./install.sh
cd ../..
```

</CollapsibleContent>

<CollapsibleContent header={<h3><span>배포 확인</span></h3>}>

Kubernetes 클러스터에 액세스할 수 있도록 로컬 kubeconfig를 업데이트합니다

:::info
AWS_REGION을 설정하지 않은 경우 아래 명령에 --region us-east-1을 사용하세요
:::

```bash
aws eks  update-kubeconfig --name aibrix-on-eks
```

먼저 클러스터에서 워커 노드가 실행 중인지 확인합니다.

```bash
kubectl get nodes
```

```bash
NAME                             STATUS   ROLES    AGE   VERSION
ip-100-64-139-184.ec2.internal   Ready    <none>   96m   v1.32.1-eks-5d632ec
ip-100-64-63-169.ec2.internal    Ready    <none>   96m   v1.32.1-eks-5d632ec
```

다음으로 모든 aibrix 파드가 실행 중인지 확인합니다.

``` bash
kubectl get pods -n aibrix-system
```

```bash
NAME                                         READY   STATUS    RESTARTS   AGE
aibrix-controller-manager-5948f8f8b7-pqwjn   1/1     Running   0          83m
aibrix-gateway-plugins-5978d98445-mrgdt      1/1     Running   0          83m
aibrix-gpu-optimizer-64c978ddd8-944mp        1/1     Running   0          83m
aibrix-kuberay-operator-8b65d7cc4-xw6bd      1/1     Running   0          83m
aibrix-metadata-service-5499dc64b7-q6rfc     1/1     Running   0          83m
aibrix-redis-master-576767646c-lqdkk         1/1     Running   0          83m
```

```bash
kubectl get deployments -A
```

```bash
NAMESPACE              NAME                                                 READY   UP-TO-DATE   AVAILABLE   AGE
aibrix-system          aibrix-controller-manager                            1/1     1            1           11m
aibrix-system          aibrix-gateway-plugins                               1/1     1            1           11m
aibrix-system          aibrix-gpu-optimizer                                 1/1     1            1           11m
aibrix-system          aibrix-kuberay-operator                              1/1     1            1           11m
aibrix-system          aibrix-metadata-service                              1/1     1            1           10m
aibrix-system          aibrix-redis-master                                  1/1     1            1           11m
envoy-gateway-system   envoy-aibrix-system-aibrix-eg-903790dc               1/1     1            1           11m
envoy-gateway-system   envoy-gateway                                        1/1     1            1           12m
ingress-nginx          ingress-nginx-controller                             1/1     1            1           11m
karpenter              karpenter                                            2/2     2            2           99m
kube-system            aws-load-balancer-controller                         2/2     2            2           12m
kube-system            coredns                                              2/2     2            2           102m
kube-system            ebs-csi-controller                                   2/2     2            2           80m
kube-system            k8s-neuron-scheduler                                 1/1     1            1           12m
kube-system            my-scheduler                                         1/1     1            1           12m
nvidia-device-plugin   nvidia-device-plugin-node-feature-discovery-master   1/1     1            1           12m
```

:::info

EKS에서 AI 모델을 배포하려면 [AIBrix Infrastructure](https://awslabs.github.io/ai-on-eks/docs/blueprints) 페이지를 참조하세요.

:::

</CollapsibleContent>

<CollapsibleContent header={<h3><span>정리</span></h3>}>

:::caution
AWS 계정에 원치 않는 요금이 청구되지 않도록 이 배포 중에 생성된 모든 AWS 리소스를 삭제하세요.
:::

이 스크립트는 `-target` 옵션을 사용하여 모든 리소스가 올바른 순서로 삭제되도록 환경을 정리합니다.

```bash
cd ai-on-eks/infra/aibrix/terraform/_LOCAL
./cleanup.sh
```

</CollapsibleContent>
