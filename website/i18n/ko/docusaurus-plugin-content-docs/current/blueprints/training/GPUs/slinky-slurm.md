---
sidebar_label: EKS에서의 Slurm
---

import CollapsibleContent from '@site/src/components/CollapsibleContent';

# EKS에서의 Slurm

:::warning
EKS에서 ML 모델을 배포하려면 GPU 또는 Neuron 인스턴스에 대한 접근 권한이 필요합니다. 배포가 작동하지 않는 경우 이러한 리소스에 대한 접근 권한이 없기 때문인 경우가 많습니다. 또한 일부 배포 패턴은 Karpenter 자동 스케일링과 정적 노드 그룹에 의존합니다. 노드가 초기화되지 않으면 Karpenter 또는 노드 그룹의 로그를 확인하여 문제를 해결하세요.
:::

### Slurm이란?

[Slurm](https://slurm.schedmd.com/overview.html)은 모든 규모의 컴퓨트 클러스터에서 컴퓨팅 리소스를 관리하기 위해 설계된 오픈소스의 고도로 확장 가능한 워크로드 관리자 및 작업 스케줄러입니다. 세 가지 핵심 기능을 제공합니다: 컴퓨팅 리소스에 대한 접근 할당, 병렬 컴퓨팅 작업을 시작하고 모니터링하기 위한 프레임워크 제공, 리소스 경합을 해결하기 위한 대기 중인 작업의 큐 관리.

Slurm은 AI 훈련에서 고성능 컴퓨팅 클러스터 전체에 걸쳐 대규모 GPU 가속 워크로드를 관리하고 스케줄링하는 데 널리 사용됩니다. 연구자와 엔지니어가 CPU, GPU, 메모리를 포함한 컴퓨팅 리소스를 효율적으로 할당할 수 있게 하며, 리소스 유형과 작업 우선순위에 대한 세밀한 제어로 여러 노드에 걸쳐 딥러닝 모델과 대규모 언어 모델의 분산 훈련을 가능하게 합니다. Slurm의 안정성, 고급 스케줄링 기능, 온프레미스 및 클라우드 환경과의 통합은 현대 AI 연구 및 산업이 요구하는 규모, 처리량, 재현성을 처리하는 데 선호되는 선택이 됩니다.

### Slinky 프로젝트란?

[Slinky 프로젝트](https://github.com/SlinkyProject)는 Slurm의 주요 개발사인 [SchedMD](https://www.schedmd.com/)가 설계한 오픈소스 통합 도구 모음으로, Slurm 기능을 Kubernetes에 도입하여 효율적인 리소스 관리와 스케줄링을 위한 두 세계의 장점을 결합합니다. Slinky 프로젝트에는 [Slurm 클러스터용 Kubernetes 오퍼레이터](https://github.com/SlinkyProject/slurm-operator?tab=readme-ov-file#kubernetes-operator-for-slurm-clusters)가 포함되어 있으며, Kubernetes 환경 내에서 배포된 Slurm Cluster 및 NodeSet 리소스의 수명 주기를 관리하기 위한 [커스텀 컨트롤러](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/#custom-controllers)와 [커스텀 리소스 정의(CRD)](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/#customresourcedefinitions)를 구현합니다.

이 Slurm 클러스터에는 다음 구성 요소가 포함됩니다:
| 구성 요소 | 설명 |
|-----------|-------------|
| Controller (slurmctld) | 리소스를 모니터링하고, 작업을 수락하며, 컴퓨트 노드에 작업을 할당하는 중앙 관리 데몬입니다. |
| Accounting (slurmdbd) | MariaDB 데이터베이스 백엔드를 통해 작업 회계 및 사용자/프로젝트 관리를 처리합니다. |
| Compute (slurmd) | 작업을 실행하는 워커 노드로, 다른 파티션으로 그룹화할 수 있는 NodeSet으로 구성됩니다. |
| Login | 사용자가 Slurm 클러스터와 상호 작용하고 작업을 제출할 수 있도록 SSH 접근 지점을 제공합니다. |
| REST API (slurmrestd) | 클러스터와의 프로그래밍 방식 상호 작용을 위해 Slurm 기능에 대한 HTTP 기반 API 접근을 제공합니다. |
| Authentication (sackd) | Slurm 서비스에 대한 안전한 접근을 위한 자격 증명 인증을 관리합니다. |
| MariaDB | 작업, 사용자 및 프로젝트 정보를 저장하기 위해 accounting 서비스에서 사용하는 데이터베이스 백엔드입니다. |
| Prometheus Service Monitor | 모니터링 목적으로 스케줄러, 파티션, 노드 및 작업 엔드포인트에서 메트릭을 수집하도록 컨트롤러 내에 구성됩니다. |

Amazon EKS와 결합하면 Slinky 프로젝트는 Kubernetes에서 인프라 관리를 표준화한 기업이 ML 과학자들에게 Slurm 기반 경험을 제공할 수 있도록 합니다. 또한 동일한 가속 노드 클러스터에서 훈련, 실험 및 추론이 이루어질 수 있게 합니다.

### EKS에서의 Slurm 아키텍처

![alt text](../img/Slurm-on-EKS.png)

위 다이어그램은 이 가이드에 설명된 EKS에서의 Slurm 배포를 보여줍니다. Amazon EKS 클러스터가 오케스트레이션 레이어 역할을 하며, 핵심 Slurm Cluster 구성 요소는 m6i.xlarge 인스턴스의 관리형 노드 그룹에서 호스팅되고, Karpenter NodePool은 slurmd 파드가 실행될 GPU 가속 컴퓨트 노드의 배포를 관리합니다. Slinky Slurm 오퍼레이터와 Slurm 클러스터는 ArgoCD 애플리케이션으로 자동 배포됩니다.

로그인 LoadBalancer 타입 서비스는 [AWS Load Balancer Controller](https://github.com/kubernetes-sigs/aws-load-balancer-controller)를 사용하여 AWS Network Load Balancer를 동적으로 생성하도록 어노테이션되어 있어, ML 과학자들이 kubectl을 통해 Kubernetes API 서버와 인터페이스하지 않고도 로그인 파드에 SSH로 접속할 수 있습니다.

로그인 및 slurmd 파드에는 [Amazon FSx for Lustre](https://aws.amazon.com/fsx/lustre/) 공유 파일 시스템도 마운트되어 있습니다. 컨테이너화된 slurmd 파드를 사용하면 전통적으로 Conda나 Python 가상 환경을 사용하여 수동으로 설치했던 많은 종속성을 컨테이너 이미지에 포함할 수 있지만, 공유 파일 시스템은 훈련 아티팩트, 데이터, 로그 및 체크포인트 저장에 여전히 유용합니다.

### 주요 기능 및 이점

- 동일한 인프라에서 Slurm 워크로드와 컨테이너화된 Kubernetes 애플리케이션을 나란히 실행합니다. Slurm과 Kubernetes 워크로드 모두 동일한 노드 풀에서 스케줄링되어 활용도를 높이고 리소스 단편화를 방지합니다.
- 양쪽 에코시스템의 익숙한 도구를 활용하여 Slurm 작업과 Kubernetes 파드를 원활하게 관리하면서 제어나 성능을 희생하지 않습니다.
- 워크로드 수요에 따라 컴퓨트 노드를 동적으로 추가하거나 제거하여 할당된 리소스를 효율적으로 자동 스케일링하고, 수요의 급증과 감소를 처리하여 인프라 비용과 유휴 리소스 낭비를 줄입니다.
- Kubernetes 오케스트레이션을 통한 고가용성. 컨트롤러나 워커 파드가 실패하면 Kubernetes가 자동으로 다시 시작하여 수동 개입을 줄입니다.
- Slurm의 정교한 스케줄링 기능(공정 공유 할당, 종속성 관리, 우선순위 스케줄링)이 Kubernetes에 통합되어 컴퓨트 활용률을 극대화하고 워크로드 요구사항에 맞춰 리소스를 조정합니다.
- Slurm과 그 종속성이 컨테이너로 배포되어 환경 전반에 걸쳐 일관된 배포를 보장합니다. 이는 구성 드리프트를 줄이고 개발에서 프로덕션으로의 전환을 간소화합니다.
- 사용자는 특수한 요구사항(예: 커스텀 종속성, 라이브러리)에 맞춤화된 Slurm 이미지를 빌드하여 과학적 또는 규제 환경에서 일관성과 재현성을 촉진할 수 있습니다.
- 관리자는 Kubernetes Custom Resources를 사용하여 커스텀 Slurm 클러스터와 노드 세트를 직접 정의할 수 있으며, 다른 유형의 작업(예: 안정적 vs 기회주의적/백필 파티션)을 위해 컴퓨트 노드를 파티셔닝할 수 있습니다.
- Slinky는 Slurm과 Kubernetes 모두를 위한 모니터링 스택과 통합되어 관리자와 사용자에게 강력한 메트릭과 시각화를 제공합니다.

<CollapsibleContent header={<h2><span>솔루션 배포</span></h2>}>

이 예제에서는 Amazon EKS에서 Slinky Slurm 클러스터를 프로비저닝합니다.

**0. 사전 요구사항:**

머신에 다음 도구가 설치되어 있는지 확인하세요.

1. [aws cli](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
2. [kubectl](https://Kubernetes.io/docs/tasks/tools/)
3. [terraform](https://learn.hashicorp.com/tutorials/terraform/install-cli)
4. [docker](https://docs.docker.com/engine/install/)
5. [helm](https://helm.sh/)

**1. 저장소 클론:**

```bash
git clone https://github.com/awslabs/ai-on-eks.git
```

:::info
인증에 프로필을 사용하는 경우
`export AWS_PROFILE="<PROFILE_name>"`을 원하는 프로필 이름으로 설정하세요
:::

**2. 구성 검토 및 커스터마이징:**

- `infra/base/terraform/variables.tf`에서 사용 가능한 애드온 확인
- 필요에 따라 `infra/slinky-slurm/terraform/blueprint.tfvars`에서 애드온 설정 수정

**3. slurmd 컨테이너 이미지 빌드 자동화 검토:**

기본적으로 `blueprints/training/slinky-slurm/install.sh` 스크립트는 `blueprints/training/slinky-slurm/dlc-slurmd.Dockerfile`을 사용하여 새 slurmd 컨테이너 이미지를 자동으로 빌드하는 설정 단계를 트리거합니다. 이 Dockerfile은 [AWS Deep Learning Container (DLC)](https://github.com/aws/deep-learning-containers) 위에 빌드되어 Python 3.12.8 + PyTorch 2.6.0 + CUDA 12.6 + NCCL 2.23.4 + EFA Installer 1.38.0 (OFI NCCL 플러그인 번들 포함)이 컨테이너 이미지에 사전 설치되어 있습니다.

그런 다음 새 ECR 저장소를 생성하고 이 이미지를 저장소에 푸시합니다. 머신에 아직 없는 경우 Slurm 로그인 파드 접근을 위한 새 SSH 키 `~/.ssh/id_ed25519_slurm`도 생성됩니다.

이미지 저장소 URI, 이미지 태그 및 공개 SSH 키는 Slurm 클러스터 배포에 사용할 `blueprints/training/slinky-slurm/slurm-values.yaml.template` 파일을 기반으로 새 `slurm-values.yaml` 파일을 생성하는 데 사용됩니다.

이 동작을 커스터마이징하려면 다음 선택적 플래그를 추가할 수 있습니다:
| 구성 요소 | 설명 | 기본값 |
|-----------|-------------|-------------|
|`--repo-name`| ECR 저장소 이름 |dlc-slurmd|
|`--tag`|이미지 태그 |25.11.1-ubuntu24.04|
|`--region`| ECR 저장소의 AWS 리전 | AWS CLI 구성에서 추론되거나 `us-west-2`로 설정|
|`--skip-build`| ECR에 이미 있는 기존 이미지를 사용하는 경우 설정 | `false`|
|`--skip-setup`| 이전에 `blueprints/training/slinky-slurm/slurm-values.yaml` 파일을 생성한 경우 설정 |`false`|
|`--help`| 플래그 옵션 보기 |`false`|

예를 들어, 커스텀 slurmd 컨테이너 이미지를 이미 빌드하고 커스텀 ECR 저장소에 푸시한 경우 다음 플래그와 값을 추가하세요:
```bash
cd ai-on-eks/blueprints/training/slinky-slurm
./install.sh --repo-name dlc-slurmd --tag 25.11.1-ubuntu24.04 --skip-build
```
스크립트는 진행하기 전에 컨테이너 이미지가 ECR 저장소에 있는지 확인합니다.

커스텀 Dockerfile을 사용하려면 `blueprints/training/slinky-slurm/install.sh`를 실행하기 전에 `blueprints/training/slinky-slurm/dlc-slurmd.Dockerfile`의 내용을 덮어쓰면 됩니다.

Terraform 배포를 트리거하지 않고 컨테이너 이미지를 빌드하고 푸시하려면 동일한 플래그를 사용하여 `blueprints/training/slinky-slurm/setup.sh` 스크립트를 직접 실행할 수도 있습니다. 이 스크립트는 새 `blueprints/training/slinky-slurm/slurm-values.yaml` 파일도 생성합니다.

**4. 배포 트리거:**

`slinky-slurm` 디렉토리로 이동하고 `install.sh` 스크립트를 실행합니다:
```bash
cd ai-on-eks/blueprints/training/slinky-slurm
./install.sh
```
</CollapsibleContent>

<CollapsibleContent header={<h3><span>배포 확인</span></h3>}>

**0. Slurm 배포를 위한 Kubernetes 리소스 확인:**

로컬 kubeconfig를 업데이트하여 kubernetes 클러스터에 접근합니다:
```
aws eks update-kubeconfig --name slurm-on-eks
```

Slinky Slurm Operator의 배포 상태 확인:
```
kubectl get all -n slinky
```
```
NAME                                         READY   STATUS    RESTARTS   AGE
pod/slurm-operator-bb5c58dc6-5rsjg           1/1     Running   0          41m
pod/slurm-operator-webhook-87bc59884-vw8rx   1/1     Running   0          41m

NAME                             TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)             AGE
service/slurm-operator           ClusterIP   None             <none>        8080/TCP,8081/TCP   41m
service/slurm-operator-webhook   ClusterIP   172.20.229.194   <none>        443/TCP,8081/TCP    41m

NAME                                     READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/slurm-operator           1/1     1            1           41m
deployment.apps/slurm-operator-webhook   1/1     1            1           41m

NAME                                               DESIRED   CURRENT   READY   AGE
replicaset.apps/slurm-operator-bb5c58dc6           1         1         1       41m
replicaset.apps/slurm-operator-webhook-87bc59884   1         1         1       41m
```

Slurm Cluster의 배포 상태 확인:
```
kubectl get all -n slurm
```
```
NAME                                      READY   STATUS    RESTARTS   AGE
pod/mariadb-0                             1/1     Running   0          9m12s
pod/slurm-accounting-0                    1/1     Running   0          9m6s
pod/slurm-controller-0                    3/3     Running   0          9m5s
pod/slurm-login-slinky-65cdfdb557-2869l   1/1     Running   0          9m5s
pod/slurm-restapi-5c6d784dbc-6nlcw        1/1     Running   0          9m5s
pod/slurm-worker-slinky-0                 2/2     Running   0          9m5s
pod/slurm-worker-slinky-1                 2/2     Running   0          9m5s
pod/slurm-worker-slinky-2                 2/2     Running   0          9m5s
pod/slurm-worker-slinky-3                 2/2     Running   0          9m5s

NAME                          TYPE           CLUSTER-IP       EXTERNAL-IP                                                                  PORT(S)        AGE
service/mariadb               ClusterIP      172.20.165.187   <none>                                                                       3306/TCP       9m12s
service/mariadb-internal      ClusterIP      None             <none>                                                                       3306/TCP       9m12s
service/slurm-accounting      ClusterIP      172.20.206.252   <none>                                                                       6819/TCP       9m6s
service/slurm-controller      ClusterIP      172.20.22.142    <none>                                                                       6817/TCP       9m5s
service/slurm-login-slinky    LoadBalancer   172.20.155.154   k8s-slurm-slurmlog-d3c664afd2-5c3621e9c562ee2d.elb.us-west-2.amazonaws.com   22:31787/TCP   9m5s
service/slurm-restapi         ClusterIP      172.20.130.229   <none>                                                                       6820/TCP       9m5s
service/slurm-workers-slurm   ClusterIP      None             <none>                                                                       6818/TCP       9m5s

NAME                                 READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/slurm-login-slinky   1/1     1            1           9m6s
deployment.apps/slurm-restapi        1/1     1            1           9m6s

NAME                                            DESIRED   CURRENT   READY   AGE
replicaset.apps/slurm-login-slinky-65cdfdb557   1         1         1       9m6s
replicaset.apps/slurm-restapi-5c6d784dbc        1         1         1       9m6s

NAME                                READY   AGE
statefulset.apps/mariadb            1/1     9m13s
statefulset.apps/slurm-accounting   1/1     9m7s
statefulset.apps/slurm-controller   1/1     9m6s
```
**1. Slurm 로그인 파드 접근:**

로그인 파드에 SSH 접속:
:::info
이 데모에서는 `slurm-login-slinky` 서비스가 `service.beta.kubernetes.io/load-balancer-source-ranges`를 사용하여 동적으로 어노테이션되어 사용자의 IP 주소만 Network Load Balancer에 대한 접근을 제한합니다. AWS Load Balancer Controller는 인바운드 보안 그룹 규칙을 수정하여 이를 달성합니다. 자세한 내용은 [접근 제어 어노테이션](https://kubernetes-sigs.github.io/aws-load-balancer-controller/v2.13/guide/ingress/annotations/#access-control) 문서를 참조하세요.
:::
```
SLURM_LOGIN_HOSTNAME="$(kubectl get svc slurm-login-slinky -n slurm \
 -o jsonpath="{.status.loadBalancer.ingress[0].hostname}")"

ssh -i ~/.ssh/id_ed25519_slurm -p 22 root@$SLURM_LOGIN_HOSTNAME
```
사용 가능한 노드 확인:
```
sinfo
```
```
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
slinky       up   infinite      4   idle slinky-[0-3]
all*         up   infinite      4   idle slinky-[0-3]
```
Amazon FSx for Lustre 공유 파일 시스템이 로그인 파드에 마운트되었는지 확인:
```
df -h
```
```
Filesystem               Size  Used Avail Use% Mounted on
overlay                   20G  8.9G   12G  45% /
tmpfs                     64M     0   64M   0% /dev
/dev/root                307M  307M     0 100% /usr/local/sbin/modprobe
10.1.0.61@tcp:/we427b4v  2.3T   16M  2.3T   1% /fsx
/dev/nvme1n1p1            20G  8.9G   12G  45% /etc/hosts
tmpfs                     15G  4.0K   15G   1% /etc/slurm
tmpfs                     15G  4.0K   15G   1% /run/slurm
shm                       64M     0   64M   0% /dev/shm
tmpfs                     15G   24K   15G   1% /etc/ssh/ssh_host_rsa_key
tmpfs                     15G  8.0K   15G   1% /etc/ssh/sshd_config
tmpfs                     15G  4.0K   15G   1% /etc/sssd/sssd.conf
tmpfs                    7.7G     0  7.7G   0% /proc/acpi
tmpfs                    7.7G     0  7.7G   0% /sys/firmware
```
머신으로 돌아가기:
```
exit
```
**2. Slurm 컴퓨트 파드 접근:**

slurm 컴퓨트 노드 중 하나와 대화형 터미널 세션 열기:
```
kubectl -n slurm exec -it pod/slurm-worker-slinky-0 -- bash --login
```
Amazon FSx for Lustre 공유 파일 시스템이 로그인 파드에 마운트되었는지 확인:
```
df -h
```
```
Filesystem               Size  Used Avail Use% Mounted on
overlay                  3.5T   95G  3.4T   3% /
tmpfs                     64M     0   64M   0% /dev
10.1.0.61@tcp:/we427b4v  2.3T   16M  2.3T   1% /fsx
tmpfs                    366G  4.0K  366G   1% /etc/slurm
tmpfs                    187G     0  187G   0% /dev/shm
/dev/nvme2n1             3.5T   95G  3.4T   3% /etc/hosts
tmpfs                    366G     0  366G   0% /var/log/slurm
tmpfs                     75G  4.4M   75G   1% /run/nvidia-persistenced/socket
/dev/root                905M  905M     0 100% /usr/bin/nvidia-smi
/dev/nvme0n1p1           300G  6.1G  294G   3% /var/lib/dcgm-exporter/job-mapping
```
컴퓨트 노드 파드에 설치된 CUDA 컴파일러 버전 확인:
```
nvcc --version
```
```
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2024 NVIDIA Corporation
# Built on Tue_Oct_29_23:50:19_PDT_2024
# Cuda compilation tools, release 12.6, V12.6.85
# Build cuda_12.6.r12.6/compiler.35059454_0
```
컴퓨트 노드 파드의 NCCL 버전 확인:
```
ldconfig -v | grep "libnccl.so" | tail -n1 | sed -r 's/^.*\.so\.//'
```
```
# 2.23.4
```
EFA 가용성 확인:
```
fi_info -p efa
```
```
provider: efa
    fabric: efa
    domain: rdmap0s26-rdm
    version: 122.0
    type: FI_EP_RDM
    protocol: FI_PROTO_EFA
provider: efa
    fabric: efa
    domain: rdmap0s26-dgrm
    version: 122.0
    type: FI_EP_DGRAM
    protocol: FI_PROTO_EFA
```
libfabric 라이브러리 확인:
```
ls /opt/amazon/efa/lib
```
```
libfabric.a  libfabric.so  libfabric.so.1  libfabric.so.1.25.0  pkgconfig
```
OFI NCCL 플러그인 라이브러리 확인:
```
ls /opt/amazon/ofi-nccl/lib/x86_64-linux-gnu
```
```
libnccl-net.so  libnccl-ofi-tuner.so
```
노드 내 GPU 토폴로지 확인:
```
nvidia-smi topo -m
```
```
	GPU0	CPU Affinity	NUMA Affinity	GPU NUMA ID
GPU0	 X 	0-31	0		N/A
```
머신으로 돌아가기:
```
exit
```
</CollapsibleContent>

<CollapsibleContent header={<h3><span>FSDP 예제 실행</span></h3>}>

**0. 훈련 아티팩트 준비:**

로그인 파드에 SSH 접속:
```
SLURM_LOGIN_HOSTNAME="$(kubectl get svc slurm-login-slinky -n slurm \
 -o jsonpath="{.status.loadBalancer.ingress[0].hostname}")"

ssh -i ~/.ssh/id_ed25519_slurm -p 22 root@$SLURM_LOGIN_HOSTNAME
```
로그인 파드에 Git 설치:
```
apt update
apt install -y git
git --version
```
FSx 마운트로 디렉토리 변경:
```
cd /fsx
```
[awsome-distributed-training](https://github.com/aws-samples/awsome-distributed-training) 저장소 클론:
```
git clone https://github.com/aws-samples/awsome-distributed-training/
```
Slurm용 FSDP 예제 디렉토리로 변경:
```
cd awsome-distributed-training/3.test_cases/pytorch/FSDP/slurm
```
체크포인트를 위한 새 디렉토리 생성
```
mkdir -p checkpoints
```
:::info
기본적으로 `llama2_7b-training.sbatch` 배치 훈련 스크립트는 4개의 `g5.8xlarge` 노드에 FSDP 워크로드를 분산하도록 구성되어 있습니다. Slurm NodeSet은 `karpenter.sh/nodepool` nodeSelector를 통해 `g5-nvidia` NodePool에 매핑됩니다. Karpenter가 4개의 온디맨드 또는 스팟 인스턴스를 찾을 수 없는 경우 배치 훈련 스크립트를 조정한 다음 프로비저닝된 S3 버킷에 새 복사본을 업로드해야 할 수 있습니다. 이는 [데이터 저장소 연결](https://docs.aws.amazon.com/fsx/latest/LustreGuide/create-dra-linked-data-repo.html)을 통해 FSx for Lustre 파일 시스템에 동기화됩니다:
```
cd ../../../infra/slinky-slurm/terraform/_LOCAL
S3_BUCKET_NAME=$(terraform output -raw fsx_s3_bucket_name)
cd ../../../../blueprints/training/slinky-slurm
aws s3 cp llama2_7b-training.sbatch s3://${S3_BUCKET_NAME}/
```
:::

`llama2_7b-training.sbatch` 배치 훈련 스크립트를 Slurm용 FSDP 예제 디렉토리로 복사:
```
cp /fsx/data/llama2_7b-training.sbatch ./llama2_7b-training.sbatch
```
**1. Hugging Face 접근 토큰 구성:**

스로틀링 없이 [allenai/c4](https://huggingface.co/datasets/allenai/c4) 데이터셋을 스트리밍하기 위해 새 [Hugging Face](https://huggingface.co/) 읽기 [사용자 접근 토큰](https://huggingface.co/docs/hub/en/security-tokens)을 생성합니다.

새 Hugging Face 토큰을 훈련 스크립트에 주입:
```
NEW_TOKEN="<you-token-here>"
```
```
sed -i "s/export HF_TOKEN=.*$/export HF_TOKEN=$NEW_TOKEN/" llama2_7b-training.sbatch
```
**2. 컨테이너화된 환경을 위한 DataLoader 구성:**

이 Kubernetes 기반 Slurm 설정에서 실행할 때 DataLoader는 기본 fork 방식 대신 spawn 기반 멀티프로세싱을 사용해야 합니다. 이는 컨테이너화된 워커 프로세스 내에서 적절한 CUDA 초기화를 보장합니다. 이를 위해 `train_utils.py` 파일에 정의된 DataLoader에 `multiprocessing_context='spawn'`을 추가합니다.

```
sed -i "s/timeout=600)/timeout=600, multiprocessing_context='spawn')/" ../src/model_utils/train_utils.py
```
**3. 훈련 작업 시작:**

[sbatch](https://slurm.schedmd.com/sbatch.html) 명령을 사용하여 Slurm Controller에 배치 훈련 스크립트 제출:
```
sbatch llama2_7b-training.sbatch
```
```
Submitted batch job 1
```
**4. 훈련 진행 상황 모니터링:**

로그인 파드에서 출력 로그 확인:
```
export JOB_ID=$(squeue -h -u root -o "%i" | head -1)

tail -f logs/llama2_7b-FSDP_${JOB_ID}.out
```
```
2: node-0:309:309 [0] NCCL INFO AllGather: 37756928 Bytes -> Algo 1 proto 2 time 2860.208008
1: node-3:278:278 [0] NCCL INFO AllGather: opCount 1606 sendbuff 0x93fd44a00 recvbuff 0x96d44e200 count 2359808 datatype 7 op 0 root 0 comm 0x555e2c6387e0 [nranks=4] stream 0x555e2b3a4370
0: node-1:397:397 [0] NCCL INFO AllGather: opCount 1606 sendbuff 0x93fd44a00 recvbuff 0x96d44e200 count 2359808 datatype 7 op 0 root 0 comm 0x55c1eaee2a20 [nranks=4] stream 0x55c1e9c4f3d0
3: node-2:278:278 [0] NCCL INFO AllGather: opCount 1606 sendbuff 0x93fd44a00 recvbuff 0x96d44e200 count 2359808 datatype 7 op 0 root 0 comm 0x5611d45babf0 [nranks=4] stream 0x5611d3327480
1: node-3:278:278 [0] NCCL INFO AllGather: opCount 1607 sendbuff 0x940645200 recvbuff 0x96d44e200 count 2359808 datatype 7 op 0 root 0 comm 0x555e2c6387e0 [nranks=4] stream 0x555e2b3a4370
3: node-2:278:278 [0] NCCL INFO AllGather: opCount 1607 sendbuff 0x940645200 recvbuff 0x96d44e200 count 2359808 datatype 7 op 0 root 0 comm 0x5611d45babf0 [nranks=4] stream 0x5611d3327480
1: node-3:278:278 [0] NCCL INFO Broadcast: opCount 1608 sendbuff 0x302287400 recvbuff 0x302287400 count 1 datatype 4 op 0 root 0 comm 0x555e2c6387e0 [nranks=4] stream 0x555e2b3a4370
0: node-1:397:397 [0] NCCL INFO AllGather: opCount 1607 sendbuff 0x940645200 recvbuff 0x96d44e200 count 2359808 datatype 7 op 0 root 0 comm 0x55c1eaee2a20 [nranks=4] stream 0x55c1e9c4f3d0
3: node-2:278:278 [0] NCCL INFO Broadcast: opCount 1608 sendbuff 0x302287400 recvbuff 0x302287400 count 1 datatype 4 op 0 root 0 comm 0x5611d45babf0 [nranks=4] stream 0x5611d3327480
1: node-3:278:278 [0] NCCL INFO Broadcast: opCount 1609 sendbuff 0x302287600 recvbuff 0x302287600 count 3145 datatype 1 op 0 root 0 comm 0x555e2c6387e0 [nranks=4] stream 0x555e2b3a4370
```
`slurm-worker-slinky-0`에서 배치 및 손실 확인 (새 터미널 창에서):
```
kubectl -n slurm exec -it pod/slurm-worker-slinky-0 -- bash --login
```
```
cd /fsx/awsome-distributed-training/3.test_cases/pytorch/FSDP/slurm
export JOB_ID=$(squeue -h -u root -o "%i" | head -1)

watch "grep 'Batch.*Loss' logs/llama2_7b-FSDP_${JOB_ID}.out"
```
```
2: 2026-01-16 06:33:23,797 [INFO] __main__: Batch 0 Loss: 11.07265, Speed: 1.17 samples/sec, lr: 0.000031
2: 2026-01-16 06:33:24,456 [INFO] __main__: Batch 1 Loss: 11.05403, Speed: 6.08 samples/sec, lr: 0.000063
2: 2026-01-16 06:33:25,111 [INFO] __main__: Batch 2 Loss: 10.88060, Speed: 6.11 samples/sec, lr: 0.000094
2: 2026-01-16 06:33:25,765 [INFO] __main__: Batch 3 Loss: 10.60247, Speed: 6.12 samples/sec, lr: 0.000100
2: 2026-01-16 06:33:26,422 [INFO] __main__: Batch 4 Loss: 10.19984, Speed: 6.09 samples/sec, lr: 0.000100
2: 2026-01-16 06:33:27,080 [INFO] __main__: Batch 5 Loss: 9.90466, Speed: 6.08 samples/sec, lr: 0.000100
2: 2026-01-16 06:33:27,733 [INFO] __main__: Batch 6 Loss: 9.80892, Speed: 6.14 samples/sec, lr: 0.000100
2: 2026-01-16 06:33:28,388 [INFO] __main__: Batch 7 Loss: 9.69072, Speed: 6.11 samples/sec, lr: 0.000100
2: 2026-01-16 06:33:29,046 [INFO] __main__: Batch 8 Loss: 9.57264, Speed: 6.09 samples/sec, lr: 0.000100
2: 2026-01-16 06:33:29,703 [INFO] __main__: Batch 9 Loss: 9.37485, Speed: 6.09 samples/sec, lr: 0.000100
2: 2026-01-16 06:33:30,357 [INFO] __main__: Batch 10 Loss: 9.25147, Speed: 6.11 samples/sec, lr: 0.000100
2: 2026-01-16 06:33:31,010 [INFO] __main__: Batch 11 Loss: 9.27672, Speed: 6.13 samples/sec, lr: 0.000100
2: 2026-01-16 06:33:31,669 [INFO] __main__: Batch 12 Loss: 9.16106, Speed: 6.08 samples/sec, lr: 0.000100
2: 2026-01-16 06:33:32,328 [INFO] __main__: Batch 13 Loss: 9.02550, Speed: 6.07 samples/sec, lr: 0.000100
2: 2026-01-16 06:33:32,981 [INFO] __main__: Batch 14 Loss: 8.93448, Speed: 6.14 samples/sec, lr: 0.000100
2: 2026-01-16 06:33:33,639 [INFO] __main__: Batch 15 Loss: 8.83249, Speed: 6.08 samples/sec, lr: 0.000100
2: 2026-01-16 06:33:34,300 [INFO] __main__: Batch 16 Loss: 8.68732, Speed: 6.05 samples/sec, lr: 0.000100
2: 2026-01-16 06:33:34,957 [INFO] __main__: Batch 17 Loss: 8.85516, Speed: 6.09 samples/sec, lr: 0.000100
2: 2026-01-16 06:33:35,616 [INFO] __main__: Batch 18 Loss: 8.63410, Speed: 6.08 samples/sec, lr: 0.000100
2: 2026-01-16 06:33:36,274 [INFO] __main__: Batch 19 Loss: 8.46158, Speed: 6.08 samples/sec, lr: 0.000100
2: 2026-01-16 06:33:36,926 [INFO] __main__: Batch 20 Loss: 8.60995, Speed: 6.14 samples/sec, lr: 0.000100
2: 2026-01-16 06:33:37,591 [INFO] __main__: Batch 21 Loss: 8.39657, Speed: 6.02 samples/sec, lr: 0.000100
2: 2026-01-16 06:33:38,246 [INFO] __main__: Batch 22 Loss: 8.26141, Speed: 6.11 samples/sec, lr: 0.000100
2: 2026-01-16 06:33:38,902 [INFO] __main__: Batch 23 Loss: 8.25075, Speed: 6.10 samples/sec, lr: 0.000100
2: 2026-01-16 06:33:39,561 [INFO] __main__: Batch 24 Loss: 8.46888, Speed: 6.08 samples/sec, lr: 0.000100
2: 2026-01-16 06:33:40,218 [INFO] __main__: Batch 25 Loss: 8.16549, Speed: 6.08 samples/sec, lr: 0.000100
2: 2026-01-16 06:33:40,870 [INFO] __main__: Batch 26 Loss: 8.17494, Speed: 6.14 samples/sec, lr: 0.000100
2: 2026-01-16 06:33:41,528 [INFO] __main__: Batch 27 Loss: 7.95398, Speed: 6.09 samples/sec, lr: 0.000100
```
`slurm-worker-slinky-1`에서 squeue 확인 (새 터미널 창에서):
```
kubectl -n slurm exec -it pod/slurm-worker-slinky-1 -- bash --login
```
```
# 1초 간격 업데이트
watch -n 1 squeue
```
```
Every 1.0s: squeue                                                                                                                                node-1: Thu Jul 17 13:18:13 2025

             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
                 1       all llama2_7     root  R       4:47      4 node-[0-3]
```
`slurm-worker-slinky-2`에서 체크포인트 확인 (새 터미널 창에서):
```
kubectl -n slurm exec -it pod/slurm-worker-slinky-2 -- bash --login
```
```
cd /fsx/awsome-distributed-training/3.test_cases/pytorch/FSDP/slurm

# 변경 사항 강조, 타임스탬프 표시, 5초 간격 업데이트
watch -n 5 -d "ls -lh checkpoints"
```
```
Every 5.0s: ls -lh checkpoints                                                                                                                    node-2: Thu Jul 17 13:35:32 2025

total 175K
drwxr-xr-x. 2 root root 25K Jul 17 13:15 llama_v2-100steps
drwxr-xr-x. 2 root root 25K Jul 17 13:16 llama_v2-200steps
drwxr-xr-x. 2 root root 25K Jul 17 13:28 llama_v2-300steps
drwxr-xr-x. 2 root root 25K Jul 17 13:29 llama_v2-400steps
drwxr-xr-x. 2 root root 25K Jul 17 13:30 llama_v2-500steps
drwxr-xr-x. 2 root root 25K Jul 17 13:32 llama_v2-600steps
drwxr-xr-x. 2 root root 25K Jul 17 13:33 llama_v2-700steps
```
머신으로 돌아가기:
```
exit
```
</CollapsibleContent>

<CollapsibleContent header={<h3><span>CloudWatch Container Insights</span></h3>}>
GPU 활용률 및 EFA 네트워크 메트릭을 보려면 [Amazon CloudWatch Container Insights](https://console.aws.amazon.com/cloudwatch/home?#container-insights:?~(query~()~context~(orchestrationService~'eks)))로 이동하세요:

![alt text](../img/GPU-Insights.png)
![alt text](../img/EFA-Insights.png)

</CollapsibleContent>

<CollapsibleContent header={<h3><span>정리</span></h3>}>

:::caution
AWS 계정에 원치 않는 요금이 발생하지 않도록 이 배포 중에 생성된 모든 AWS 리소스를 삭제하세요.
:::

이 스크립트는 모든 리소스가 올바른 순서로 삭제되도록 `-target` 옵션을 사용하여 환경을 정리합니다.

```bash
cd ai-on-eks/blueprints/training/slinky-slurm
```
```
./cleanup.sh
```
</CollapsibleContent>
