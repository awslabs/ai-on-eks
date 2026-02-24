---
sidebar_label: EKS 기반 JARK
sidebar_position: 1
---
import CollapsibleContent from '@site/src/components/CollapsibleContent';

# EKS 기반 JARK

:::warning
EKS에서 ML 모델을 배포하려면 GPU 또는 Neuron 인스턴스에 대한 액세스가 필요합니다. 배포가 작동하지 않는 경우 이러한 리소스에 대한 액세스가 누락되어 있는 경우가 많습니다. 또한 일부 배포 패턴은 Karpenter 자동 확장 및 정적 노드 그룹에 의존합니다. 노드가 초기화되지 않는 경우 Karpenter 또는 노드 그룹의 로그를 확인하여 문제를 해결하세요.
:::

:::info
이 지침은 JARK 클러스터를 기본으로만 배포합니다. 추론 또는 훈련을 위한 특정 모델을 배포하려면 [AI](https://awslabs.github.io/ai-on-eks/docs/blueprints) 페이지에서 엔드투엔드 지침을 참조하세요.
:::

### JARK란?
JARK는 [JupyterHub](https://jupyter.org/hub), [Argo Workflows](https://github.com/argoproj/argo-workflows), [Ray](https://github.com/ray-project/ray), [Kubernetes](https://kubernetes.io/)로 구성된 강력한 스택으로, Amazon EKS에서 생성형 AI 모델의 배포 및 관리를 간소화하도록 설계되었습니다. 이 스택은 AI 및 Kubernetes 에코시스템에서 가장 효과적인 도구들을 결합하여 대규모 AI 모델의 훈련, 파인튜닝 및 추론을 위한 강력한 솔루션을 제공합니다.

JARK는 [AI/ML 관측성](https://github.com/awslabs/ai-ml-observability-reference-architecture)이 활성화되어 있습니다. 관측성 아키텍처에 대한 자세한 내용은 [관측성](https://awslabs.github.io/ai-on-eks/docs/bestpractices/observability) 섹션을 참조하세요.

### 주요 기능 및 이점
[JupyterHub](https://jupyter.org/hub): 모델 개발 및 프롬프트 엔지니어링에 필수적인 노트북 실행을 위한 협업 환경을 제공합니다.

[Argo Workflows](https://github.com/argoproj/argo-workflows): 데이터 준비부터 모델 배포까지 전체 AI 모델 파이프라인을 자동화하여 일관되고 효율적인 프로세스를 보장합니다.

[Ray](https://github.com/ray-project/ray): 여러 노드에 걸쳐 AI 모델 훈련 및 추론을 확장하여 대용량 데이터셋을 더 쉽게 처리하고 훈련 시간을 단축합니다.

[Kubernetes](https://kubernetes.io/): 고가용성과 리소스 효율성으로 컨테이너화된 AI 모델을 실행, 확장 및 관리하는 데 필요한 오케스트레이션을 제공하여 스택을 구동합니다.

### JARK를 사용해야 하는 이유
JARK 스택은 AI 모델 배포 및 관리의 복잡한 프로세스를 단순화하려는 팀과 조직에 이상적입니다. 최첨단 생성형 모델을 작업하든 기존 AI 워크로드를 확장하든, Amazon EKS 기반 JARK는 성공에 필요한 유연성, 확장성 및 제어 기능을 제공합니다.


![alt text](../img/jark.png)


### Kubernetes 기반 Ray

[Ray](https://www.ray.io/)는 확장 가능하고 분산된 애플리케이션을 구축하기 위한 오픈소스 프레임워크입니다. 분산 컴퓨팅을 위한 간단하고 직관적인 API를 제공하여 병렬 및 분산 Python 애플리케이션을 쉽게 작성할 수 있도록 설계되었습니다. 사용자 및 기여자 커뮤니티가 성장하고 있으며, Anyscale, Inc.의 Ray 팀에서 적극적으로 유지 관리 및 개발하고 있습니다.

![RayCluster](../img/ray-cluster.svg)

*출처: https://docs.ray.io/en/latest/cluster/key-concepts.html*

프로덕션에서 여러 머신에 Ray를 배포하려면 사용자가 먼저 [**Ray Cluster**](https://docs.ray.io/en/latest/cluster/getting-started.html)를 배포해야 합니다. Ray Cluster는 헤드 노드와 워커 노드로 구성되며, 내장된 **Ray Autoscaler**를 사용하여 자동 확장할 수 있습니다.

Amazon EKS를 포함한 Kubernetes에서 Ray Cluster 배포는 [**KubeRay Operator**](https://ray-project.github.io/kuberay/)를 통해 지원됩니다. 이 오퍼레이터는 Ray 클러스터를 관리하는 Kubernetes 네이티브 방식을 제공합니다. KubeRay Operator 설치에는 [여기](https://ray-project.github.io/kuberay/deploy/helm/)에 문서화된 대로 `RayCluster`, `RayJob` 및 `RayService`용 오퍼레이터 및 CRD 배포가 포함됩니다.

Kubernetes에서 Ray를 배포하면 다음과 같은 여러 이점을 얻을 수 있습니다:

1. **확장성**: Kubernetes를 사용하면 워크로드 요구 사항에 따라 Ray 클러스터를 확장하거나 축소할 수 있어 대규모 분산 애플리케이션을 쉽게 관리할 수 있습니다.

1. **내결함성**: Kubernetes는 노드 장애를 처리하고 Ray 클러스터의 고가용성을 보장하는 내장 메커니즘을 제공합니다.

1. **리소스 할당**: Kubernetes를 사용하면 Ray 워크로드에 대한 리소스를 쉽게 할당하고 관리하여 최적의 성능을 위해 필요한 리소스에 액세스할 수 있습니다.

1. **이식성**: Kubernetes에서 Ray를 배포하면 여러 클라우드 및 온프레미스 데이터 센터에서 워크로드를 실행할 수 있어 필요에 따라 애플리케이션을 쉽게 이동할 수 있습니다.

1. **모니터링**: Kubernetes는 메트릭 및 로깅을 포함한 풍부한 모니터링 기능을 제공하여 문제 해결 및 성능 최적화를 쉽게 할 수 있습니다.

전반적으로 Kubernetes에서 Ray를 배포하면 분산 애플리케이션의 배포 및 관리를 단순화할 수 있어 대규모 머신 러닝 워크로드를 실행해야 하는 많은 조직에서 인기 있는 선택입니다.

배포를 진행하기 전에 공식 [문서](https://docs.ray.io/en/latest/cluster/kubernetes/index.html)의 관련 섹션을 읽어보시기 바랍니다.

![RayonK8s](../img/ray_on_kubernetes.webp)

*출처: https://docs.ray.io/en/latest/cluster/kubernetes/index.html*

<CollapsibleContent header={<h2><span>솔루션 배포</span></h2>}>

이 [예제](https://github.com/awslabs/ai-on-eks/tree/main/infra/jark-stack/terraform)에서는 Amazon EKS에 JARK 클러스터를 프로비저닝합니다.

![JARK](../img/jark-stack.png)


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
- 필요에 따라 `infra/jark-stack/terraform/blueprint.tfvars`에서 애드온 설정을 수정하세요
- `blueprint.tfvars`에서 AWS 리전을 업데이트하세요

**3. 배포 디렉터리로 이동하여 설치 스크립트를 실행합니다:**

```bash
cd ai-on-eks/infra/jark-stack && chmod +x install.sh
./install.sh
```

</CollapsibleContent>

<CollapsibleContent header={<h3><span>배포 확인</span></h3>}>

Kubernetes 클러스터에 액세스할 수 있도록 로컬 kubeconfig를 업데이트합니다

```bash
aws eks update-kubeconfig --name jark-stack #또는 EKS 클러스터 이름으로 사용한 이름
```

먼저 클러스터에서 워커 노드가 실행 중인지 확인합니다.

```bash
kubectl get nodes
```

```bash
NAME                             STATUS   ROLES    AGE     VERSION
ip-100-64-218-158.ec2.internal   Ready    <none>   3h13m   v1.32.3-eks-473151a
ip-100-64-39-78.ec2.internal     Ready    <none>   3h13m   v1.32.3-eks-473151a
```

다음으로 모든 파드가 실행 중인지 확인합니다.

```bash
kubectl get deployments -A
```

```bash
NAMESPACE              NAME                                                 READY   UP-TO-DATE   AVAILABLE   AGE
amazon-cloudwatch      amazon-cloudwatch-observability-controller-manager   1/1     1            1           3h3m
argo-events            argo-events-controller-manager                       1/1     1            1           3h2m
argo-events            events-webhook                                       1/1     1            1           3h2m
argo-workflows         argo-workflows-server                                1/1     1            1           3h2m
argo-workflows         argo-workflows-workflow-controller                   1/1     1            1           3h2m
argocd                 argocd-applicationset-controller                     1/1     1            1           3h2m
argocd                 argocd-dex-server                                    1/1     1            1           3h2m
argocd                 argocd-notifications-controller                      1/1     1            1           3h2m
argocd                 argocd-redis                                         1/1     1            1           3h2m
argocd                 argocd-repo-server                                   1/1     1            1           3h2m
argocd                 argocd-server                                        1/1     1            1           3h2m
ingress-nginx          ingress-nginx-controller                             1/1     1            1           3h1m
jupyterhub             hub                                                  1/1     1            1           3h1m
jupyterhub             proxy                                                1/1     1            1           3h1m
jupyterhub             user-scheduler                                       2/2     2            2           3h1m
karpenter              karpenter                                            2/2     2            2           3h1m
kube-system            aws-load-balancer-controller                         2/2     2            2           3h1m
kube-system            coredns                                              2/2     2            2           3h8m
kube-system            ebs-csi-controller                                   2/2     2            2           3h4m
kube-system            efs-csi-controller                                   2/2     2            2           3h2m
kube-system            k8s-neuron-scheduler                                 1/1     1            1           3h1m
kube-system            metrics-server                                       2/2     2            2           3h4m
kube-system            my-scheduler                                         1/1     1            1           3h1m
kuberay-operator       kuberay-operator                                     1/1     1            1           3h1m
monitoring             fluent-operator                                      1/1     1            1           178m
monitoring             kube-prometheus-stack-grafana                        1/1     1            1           178m
monitoring             kube-prometheus-stack-kube-state-metrics             1/1     1            1           178m
monitoring             kube-prometheus-stack-operator                       1/1     1            1           178m
monitoring             opencost                                             1/1     1            1           178m
monitoring             opensearch-dashboards                                2/2     2            2           177m
monitoring             opensearch-operator-controller-manager               1/1     1            1           178m
nvidia-device-plugin   nvidia-device-plugin-node-feature-discovery-master   1/1     1            1           23m
```

:::info

EKS에서 AI 모델을 배포하려면 [AI](https://awslabs.github.io/ai-on-eks/docs/blueprints) 페이지를 참조하세요.

:::

</CollapsibleContent>

<CollapsibleContent header={<h3><span>정리</span></h3>}>

:::caution
AWS 계정에 원치 않는 요금이 청구되지 않도록 이 배포 중에 생성된 모든 AWS 리소스를 삭제하세요.
:::

이 스크립트는 `-target` 옵션을 사용하여 모든 리소스가 올바른 순서로 삭제되도록 환경을 정리합니다.

```bash
cd ai-on-eks/infra/jark-stack/terraform && chmod +x cleanup.sh
```

</CollapsibleContent>
