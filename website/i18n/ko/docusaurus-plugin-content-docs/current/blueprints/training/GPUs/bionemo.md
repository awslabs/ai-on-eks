---
sidebar_position: 1
sidebar_label: EKS에서의 BioNeMo
---
import CollapsibleContent from '@site/src/components/CollapsibleContent';

# EKS에서의 BioNeMo

:::warning
EKS에서 ML 모델을 배포하려면 GPU 또는 Neuron 인스턴스에 대한 접근 권한이 필요합니다. 배포가 작동하지 않는 경우 이러한 리소스에 대한 접근 권한이 없기 때문인 경우가 많습니다. 또한 일부 배포 패턴은 Karpenter 자동 스케일링과 정적 노드 그룹에 의존합니다. 노드가 초기화되지 않으면 Karpenter 또는 노드 그룹의 로그를 확인하여 문제를 해결하세요.
:::

:::caution
이 블루프린트는 실험적인 것으로 간주되어야 하며 개념 증명에만 사용해야 합니다.
:::


## 소개

[NVIDIA BioNeMo](https://www.nvidia.com/en-us/clara/bionemo/)는 신약 개발을 위한 생성형 AI 플랫폼으로, 자체 데이터를 사용한 모델 훈련을 단순화하고 가속화하며 신약 개발 애플리케이션을 위한 모델 배포를 확장합니다. BioNeMo는 AI 모델 개발과 배포 모두에 가장 빠른 경로를 제공하여 AI 기반 신약 개발로의 여정을 가속화합니다. 사용자와 기여자 커뮤니티가 성장하고 있으며 NVIDIA에서 적극적으로 유지 관리하고 개발하고 있습니다.

컨테이너화된 특성을 고려할 때 BioNeMo는 Amazon Sagemaker, AWS ParallelCluster, Amazon ECS, Amazon EKS와 같은 다양한 환경에서 배포의 다양성을 찾습니다. 그러나 이 솔루션은 Amazon EKS에서의 BioNeMo 배포에 중점을 둡니다.

*출처: https://blogs.nvidia.com/blog/bionemo-on-aws-generative-ai-drug-discovery/*

## Kubernetes에서 BioNeMo 배포

이 블루프린트는 기능을 위해 세 가지 주요 구성 요소를 활용합니다. NVIDIA Device Plugin은 GPU 사용을 용이하게 하고, FSx는 훈련 데이터를 저장하며, Kubeflow Training Operator는 실제 훈련 프로세스를 관리합니다.

1) [**Kubeflow Training Operator**](https://www.kubeflow.org/docs/components/training/)
2) [**NVIDIA Device Plugin**](https://github.com/NVIDIA/k8s-device-plugin)
3) [**FSx for Lustre CSI Driver**](https://docs.aws.amazon.com/eks/latest/userguide/fsx-csi.html)


이 블루프린트에서는 Amazon EKS 클러스터를 배포하고 데이터 준비 작업과 분산 모델 훈련 작업을 모두 실행합니다.

<CollapsibleContent header={<h3><span>사전 요구사항</span></h3>}>

Mac, Windows 또는 Cloud9 IDE와 같이 Terraform 블루프린트를 배포하는 데 사용하는 로컬 머신 또는 머신에 다음 도구가 설치되어 있는지 확인하세요:

1. [aws cli](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
2. [kubectl](https://Kubernetes.io/docs/tasks/tools/)
3. [terraform](https://learn.hashicorp.com/tutorials/terraform/install-cli)

</CollapsibleContent>

<CollapsibleContent header={<h3><span>블루프린트 배포</span></h3>}>

#### 저장소 클론

먼저 블루프린트 배포에 필요한 파일이 포함된 저장소를 클론합니다. 터미널에서 다음 명령을 사용하세요:

```bash
git clone https://github.com/awslabs/ai-on-eks.git
```

#### Terraform 초기화

배포하려는 블루프린트에 해당하는 디렉토리로 이동합니다. 이 경우 BioNeMo 블루프린트에 관심이 있으므로 터미널을 사용하여 적절한 디렉토리로 이동합니다:

```bash
cd ai-on-eks/infra/bionemo
```

#### 설치 스크립트 실행

제공된 헬퍼 스크립트 `install.sh`를 사용하여 terraform init 및 apply 명령을 실행합니다. 기본적으로 스크립트는 EKS 클러스터를 `us-west-2` 리전에 배포합니다. 리전을 변경하려면 `blueprint.tfvars`를 업데이트하세요. 이 시점에 다른 입력 변수를 업데이트하거나 terraform 템플릿에 다른 변경을 가할 수도 있습니다.


```bash
./install.sh
```

kubernetes 클러스터에 접근할 수 있도록 로컬 kubeconfig를 업데이트합니다

```bash
aws eks update-kubeconfig --name bionemo-on-eks #또는 EKS 클러스터 이름으로 사용한 것
```

Training Operator용 helm 차트가 없으므로 패키지를 수동으로 설치해야 합니다. training-operator 팀에서 helm 차트를 빌드하면 terraform-aws-eks-data-addons 저장소에 통합할 것입니다.

#### Kubeflow Training Operator 설치
```bash
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.7.0"
```

</CollapsibleContent>

<CollapsibleContent header={<h3><span>배포 확인</span></h3>}>

먼저 클러스터에서 워커 노드가 실행 중인지 확인합니다.

```bash
kubectl get nodes
```
```bash
NAME                            STATUS   ROLES    AGE   VERSION
ip-100-64-229-12.ec2.internal   Ready    <none>   20m   v1.30.9-eks-5d632ec
ip-100-64-89-235.ec2.internal   Ready    <none>   20m   v1.30.9-eks-5d632ec
...
```

다음으로 모든 파드가 실행 중인지 확인합니다.

```bash
kubectl get pods -A
```

```bash
NAMESPACE              NAME                                                              READY   STATUS    RESTARTS      AGE
amazon-guardduty       aws-guardduty-agent-8tn88                                         1/1     Running   2 (21m ago)   21m
amazon-guardduty       aws-guardduty-agent-vsdmc                                         1/1     Running   1 (19m ago)   21m
ingress-nginx          ingress-nginx-controller-6c4dd4ddcc-jz7q4                         1/1     Running   1             19m
karpenter              karpenter-64db88475b-nwhbz                                        1/1     Running   0             22m
karpenter              karpenter-64db88475b-qp7np                                        1/1     Running   1 (19m ago)   22m
kube-system            aws-load-balancer-controller-6bdc9bc5cf-c6snh                     1/1     Running   0             19m
kube-system            aws-load-balancer-controller-6bdc9bc5cf-p5l5f                     1/1     Running   0             19m
kube-system            aws-node-92266                                                    2/2     Running   0             19m
kube-system            aws-node-xmpdc                                                    2/2     Running   0             19m
kube-system            coredns-6885b74c4d-6vj7d                                          1/1     Running   0             19m
kube-system            coredns-6885b74c4d-j5h74                                          1/1     Running   0             19m
kube-system            ebs-csi-controller-759d79666-45jnz                                6/6     Running   0             19m
kube-system            ebs-csi-controller-759d79666-tsrlz                                6/6     Running   0             19m
kube-system            ebs-csi-node-tfh2t                                                3/3     Running   0             19m
kube-system            ebs-csi-node-x2j9n                                                3/3     Running   0             19m
kube-system            fsx-csi-controller-64dcfcbfcb-qtwmp                               4/4     Running   0             19m
kube-system            fsx-csi-controller-64dcfcbfcb-zr2qk                               4/4     Running   0             19m
kube-system            fsx-csi-node-78mc6                                                3/3     Running   0             19m
kube-system            fsx-csi-node-q7947                                                3/3     Running   0             19m
kube-system            kube-proxy-f45kr                                                  1/1     Running   0             19m
kube-system            kube-proxy-ffk5d                                                  1/1     Running   0             19m
kubeflow               training-operator-66d8d6745f-4nr4r                                1/1     Running   0             58s
nvidia-device-plugin   nvidia-device-plugin-node-feature-discovery-master-695f7b9gk2s6   1/1     Running   0             19m
...
```
:::info
training-operator, nvidia-device-plugin 및 fsx-csi-controller 파드가 실행 중이고 정상인지 확인하세요.

:::
</CollapsibleContent>


### BioNeMo 훈련 작업 실행

모든 구성 요소가 제대로 작동하는지 확인한 후 클러스터에 작업을 제출할 수 있습니다.

#### 1단계: Uniref50 데이터 준비 작업 시작

`uniref50-job.yaml`이라는 첫 번째 작업은 처리 효율성을 높이기 위해 데이터를 다운로드하고 파티셔닝하는 것입니다. 이 작업은 특별히 `uniref50 데이터셋`을 검색하고 FSx for Lustre 파일 시스템 내에 구성합니다. 이 구조화된 레이아웃은 훈련, 테스트 및 검증 목적을 위해 설계되었습니다. uniref 데이터셋에 대한 자세한 내용은 [여기](https://www.uniprot.org/help/uniref)에서 확인할 수 있습니다.

이 작업을 실행하려면 `examples/esm1nv` 디렉토리로 이동하고 다음 명령을 사용하여 `uniref50-job.yaml` 매니페스트를 배포합니다:

```bash
cd examples/esm1nv
kubectl apply -f uniref50-job.yaml
```

:::info

이 작업은 일반적으로 50~60시간 정도의 상당한 시간이 필요합니다.

:::

아래 명령을 실행하여 `uniref50-download-*` 파드를 찾습니다

```bash
kubectl get pods
```

진행 상황을 확인하려면 해당 파드에서 생성된 로그를 검토합니다:

```bash
kubectl logs uniref50-download-xnz42

[NeMo I 2024-02-26 23:02:20 preprocess:289] Download and preprocess of UniRef50 data does not currently use GPU. Workstation or CPU-only instance recommended.
[NeMo I 2024-02-26 23:02:20 preprocess:115] Data processing can take an hour or more depending on system resources.
[NeMo I 2024-02-26 23:02:20 preprocess:117] Downloading file from https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz...
[NeMo I 2024-02-26 23:02:20 preprocess:75] Downloading file to /fsx/raw/uniref50.fasta.gz...
[NeMo I 2024-02-26 23:08:33 preprocess:89] Extracting file to /fsx/raw/uniref50.fasta...
[NeMo I 2024-02-26 23:12:46 preprocess:311] UniRef50 data processing complete.
[NeMo I 2024-02-26 23:12:46 preprocess:313] Indexing UniRef50 dataset.
[NeMo I 2024-02-26 23:16:21 preprocess:319] Writing processed dataset files to /fsx/processed...
[NeMo I 2024-02-26 23:16:21 preprocess:255] Creating train split...
```


이 작업이 완료되면 처리된 데이터셋이 `/fsx/processed` 디렉토리에 저장됩니다. 이 작업이 완료되면 다음 명령을 실행하여 `사전 훈련` 작업을 시작할 수 있습니다:

다음으로 다음을 실행하여 사전 훈련 작업을 실행할 수 있습니다:

이 PyTorchJob YAML에서 `python3 -m torch.distributed.run` 명령은 Kubernetes 클러스터의 여러 워커 파드에서 **분산 훈련**을 조율하는 데 중요한 역할을 합니다.

다음 작업을 처리합니다:

1. 워커 프로세스 간 통신을 위한 분산 백엔드(예: c10d, NCCL)를 초기화합니다. 이 예제에서는 c10d를 사용합니다. 이는 환경에 따라 TCP 또는 Infiniband와 같은 다양한 통신 메커니즘을 활용할 수 있는 PyTorch에서 일반적으로 사용되는 분산 백엔드입니다.
2. 훈련 스크립트 내에서 분산 훈련을 활성화하기 위한 환경 변수를 설정합니다.
3. 모든 워커 파드에서 훈련 스크립트를 시작하여 각 프로세스가 분산 훈련에 참여하도록 합니다.


```bash
kubectl apply -f esm1nv_pretrain-job.yaml
```

아래 명령을 실행하여 `esm1nv-pretraining-worker-*` 파드를 찾습니다

```bash
kubectl get pods
```

```bash
NAME                           READY   STATUS    RESTARTS   AGE
esm1nv-pretraining-worker-0   1/1     Running   0          11m
esm1nv-pretraining-worker-1   1/1     Running   0          11m
esm1nv-pretraining-worker-2   1/1     Running   0          11m
esm1nv-pretraining-worker-3   1/1     Running   0          11m
esm1nv-pretraining-worker-4   1/1     Running   0          11m
esm1nv-pretraining-worker-5   1/1     Running   0          11m
esm1nv-pretraining-worker-6   1/1     Running   0          11m
esm1nv-pretraining-worker-7   1/1     Running   0          11m
```

8개의 파드가 실행 중인 것을 확인해야 합니다. 파드 정의에서 각각 1개의 GPU 제한이 있는 8개의 워커 레플리카를 지정했습니다. Karpenter가 각각 4개의 GPU가 있는 2개의 g5.12xlarge 인스턴스를 프로비저닝했습니다. "nprocPerNode"를 "4"로 설정했으므로 각 노드는 4개의 작업을 담당합니다. 분산 pytorch 훈련에 대한 자세한 내용은 [pytorch 문서](https://pytorch.org/docs/stable/distributed.html)를 참조하세요.

:::info
이 훈련 작업은 g5.12xlarge 노드에서 최소 3-4일 동안 실행될 수 있습니다.
:::

이 구성은 Kubeflow의 PyTorch 훈련 Custom Resource Definition (CRD)을 활용합니다. 이 매니페스트 내에서 다양한 파라미터를 커스터마이징할 수 있습니다. 각 파라미터에 대한 자세한 인사이트와 파인튜닝 가이드는 [BioNeMo 문서](https://docs.nvidia.com/bionemo-framework/latest/notebooks/model_training_esm1nv.html)를 참조할 수 있습니다.

:::info
Kubeflow training operator 문서에 따르면 마스터 레플리카 파드를 명시적으로 지정하지 않으면 첫 번째 워커 레플리카 파드(worker-0)가 마스터 파드로 처리됩니다.
:::

이 프로세스의 진행 상황을 추적하려면 다음 단계를 따르세요:

```bash
kubectl logs esm1nv-pretraining-worker-0

Epoch 0:   7%|▋         | 73017/1017679 [00:38<08:12, 1918.0%
```

또한 해당 노드에서 실행 중인 Kubernetes 파드 내에서 `nvidia-smi` 명령을 실행하여 특정 워커 노드의 GPU 상태 스냅샷을 얻을 수 있습니다. 더 강력한 관측성을 원한다면 [DCGM Exporter](https://docs.nvidia.com/datacenter/cloud-native/gpu-telemetry/latest/dcgm-exporter.html)를 참조할 수 있습니다.


```bash
kubectl exec esm1nv-pretraining-worker-0 -- nvidia-smi
Mon Feb 24 18:51:35 2025
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.230.02             Driver Version: 535.230.02   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A10G                    On  | 00000000:00:1E.0 Off |                    0 |
|  0%   33C    P0             112W / 300W |   3032MiB / 23028MiB |     95%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
+---------------------------------------------------------------------------------------+
```


#### 분산 훈련의 이점:

여러 워커 파드의 여러 GPU에 훈련 워크로드를 분산함으로써 모든 GPU의 결합된 연산 능력을 활용하여 대규모 모델을 더 빠르게 훈련할 수 있습니다. 단일 GPU의 메모리에 맞지 않을 수 있는 더 큰 데이터셋도 처리할 수 있습니다.

#### 결론
BioNeMo는 신약 개발 영역을 위해 맞춤화된 강력한 생성형 AI 도구입니다. 이 예제에서는 광범위한 uniref50 데이터셋을 활용하여 처음부터 커스텀 모델을 사전 훈련하는 것을 주도적으로 수행했습니다. 그러나 BioNeMo는 [NVidia에서 제공하는](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/containers/bionemo-framework) 사전 훈련된 모델을 직접 사용하여 프로세스를 신속하게 진행할 수 있는 유연성을 제공한다는 점에 주목할 가치가 있습니다. 이 대안적 접근 방식은 BioNeMo 프레임워크의 강력한 기능을 유지하면서 워크플로우를 크게 간소화할 수 있습니다.


<CollapsibleContent header={<h3><span>정리</span></h3>}>

제공된 헬퍼 스크립트 `cleanup.sh`를 사용하여 EKS 클러스터 및 기타 AWS 리소스를 해제합니다.

```bash
../../terraform/_LOCAL/cleanup.sh
```

</CollapsibleContent>
