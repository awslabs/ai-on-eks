---
sidebar_position: 0
sidebar_label: 소개
---

# 소개

AI on EKS 기반 인프라는 `infra/base` 디렉토리에 있습니다. 이 디렉토리에는 실험, AI/ML 학습, LLM 추론, 모델 추적 등을 지원하는 환경을 구성할 수 있는 기본 인프라와 모든 모듈이 포함되어 있습니다.

디렉토리에는 원하는 모듈을 활성화하거나 비활성화하는 데 사용되는 모든 매개변수가 포함된 `variables.tf`가 있습니다 (기본값은 `false`로 설정). 이를 통해 Karpenter와 GPU 및 AWS Neuron `NodePool`을 갖춘 기본 환경을 배포하여 가속기 사용 및 추가 커스터마이징이 가능합니다.

참조 구현인 `jark-stack`은 실험을 위한 JupyterHub, [Ray Clusters](https://docs.ray.io/en/latest/cluster/getting-started.html)를 사용한 학습 및 추론을 위한 KubeRay 오퍼레이터, 워크플로우 자동화를 위한 Argo Workflows, 스토리지 컨트롤러 및 볼륨을 활성화하여 빠른 AI/ML 개발을 지원하는 환경을 배포합니다.

다른 블루프린트는 동일한 기반 인프라를 사용하며 블루프린트의 필요에 따라 다른 컴포넌트를 선택적으로 활성화합니다.

## 개요

AI on EKS는 Amazon EKS에서 AI/ML 워크로드를 배포하기 위한 포괄적인 인프라 솔루션을 제공합니다. 학습, 추론 또는 범용 AI/ML 워크로드에 최적화된 사전 구성 솔루션 중 선택하세요.

### 학습 인프라

AI/ML 모델 학습 워크로드에 최적화된 인프라 솔루션:

- **[JARK Stack on EKS](./training/jark.md)** - JupyterHub, Ray, Kubeflow를 포함한 NVIDIA GPU 기반 AI 워크로드를 위한 완전한 스택
- **[JupyterHub on EKS](./training/jupyterhub.md)** - 데이터 사이언스 및 ML을 위한 대화형 개발 환경

### 추론 인프라

AI/ML 모델 추론 워크로드에 최적화된 인프라 솔루션:

- **[추론 준비 클러스터](./inference/inference-ready-cluster.md)** - 추론 워크로드를 위해 사전 구성된 EKS 클러스터
- **[Nvidia NIM on EKS](../blueprints/inference/framework-guides/GPUs/nvidia-nim-llama3.md)** - Nvidia NIM 배포 샘플
- **[Nvidia Dynamo on EKS](../blueprints/inference/framework-guides/GPUs/nvidia-dynamo.md)** - Nvidia Dynamo 배포 샘플

### 기타

추가 인프라 솔루션 및 유틸리티:

- **[EMR Spark Rapids](./misc/emr-spark-rapids.md)** - Amazon EMR에서 GPU 가속 Apache Spark
- **[문제 해결](./misc/troubleshooting/troubleshooting.md)** - 일반적인 문제 및 솔루션

### 시작하기

1. **사용 사례 선택**: 워크로드 요구 사항에 따라 학습 또는 추론 선택
2. **인프라 배포**: 선택한 솔루션의 배포 가이드를 따라 진행
3. **워크로드 배포**: [블루프린트](../blueprints/index.md)를 사용하여 AI/ML 워크로드 배포
4. **최적화**: [가이던스](../guidance/index.md) 모범 사례 적용

### 아키텍처 패턴

모든 인프라 솔루션은 다음 핵심 원칙을 따릅니다:

- **모듈식 설계**: 재사용 가능한 모듈로 솔루션 구성
- **모범 사례**: 보안, 관측성, 확장성이 내장
- **클라우드 네이티브**: Kubernetes와 AWS 서비스 활용
- **검증됨**: 엔터프라이즈 워크로드에 대해 테스트 및 검증 완료

## 리소스

각 스택은 `base` 스택의 컴포넌트를 상속합니다. 이러한 컴포넌트에는 다음이 포함됩니다:

- 2개 가용 영역에 서브넷이 있는 VPC
- 최소 인프라를 실행하기 위한 2개 노드를 가진 1개 코어 노드그룹이 있는 EKS 클러스터
- CPU, GPU, AWS Neuron NodePool을 갖춘 Karpenter 오토스케일링
- GPU/Neuron 디바이스 드라이버
- GPU/Neuron 모니터링 에이전트

## 변수

### 배포

| 변수 이름                                | 설명                                                | 기본값                   |
|------------------------------------------|-----------------------------------------------------|--------------------------|
| `name`                                   | Kubernetes 클러스터 이름                            | `ai-stack`               |
| `region`                                 | 클러스터 리전                                       | us-east-1                |
| `eks_cluster_version`                    | 사용할 EKS 버전                                     | 1.32                     |
| `vpc_cidr`                               | VPC에 사용되는 CIDR                                 | `10.1.0.0/21`            |
| `secondary_cidr_blocks`                  | VPC 보조 CIDR                                       | `100.64.0.0/16`          |
| `enable_database_subnets`                | 데이터베이스 서브넷 활성화 여부                     | `false`                  |
| `enable_aws_cloudwatch_metrics`          | AWS CloudWatch Metrics 애드온 활성화                | `false`                  |
| `bottlerocket_data_disk_snapshot_id`     | 배포된 노드에 스냅샷 ID 연결                        | `""`                     |
| `enable_aws_efs_csi_driver`              | AWS EFS CSI 드라이버 활성화                         | `false`                  |
| `enable_aws_efa_k8s_device_plugin`       | AWS EFA 디바이스 플러그인 활성화                    | `false`                  |
| `enable_aws_fsx_csi_driver`              | FSx 디바이스 플러그인 활성화                        | `false`                  |
| `deploy_fsx_volume`                      | 간단한 FSx 볼륨 배포                               | `false`                  |
| `fsx_pvc_namespace`                      | FSx PVC를 프로비저닝할 네임스페이스                 | `default`                |
| `enable_amazon_prometheus`               | Amazon Managed Prometheus 활성화                    | `false`                  |
| `enable_amazon_emr`                      | Amazon EMR 설정                                     | `false`                  |
| `enable_kube_prometheus_stack`           | Kube Prometheus 애드온 활성화                       | `false`                  |
| `enable_kubecost`                        | Kubecost 활성화                                     | `false`                  |
| `enable_ai_ml_observability_stack`       | AI/ML 관측성 애드온 활성화                          | `false`                  |
| `enable_argo_workflows`                  | Argo Workflow 활성화                                | `false`                  |
| `enable_argo_events`                     | Argo Events 활성화                                  | `false`                  |
| `enable_argocd`                          | ArgoCD 애드온 활성화                                | `false`                  |
| `enable_mlflow_tracking`                 | MLFlow Tracking 활성화                              | `false`                  |
| `enable_jupyterhub`                      | JupyterHub 활성화                                   | `false`                  |
| `enable_volcano`                         | Volcano 활성화                                      | `false`                  |
| `enable_kuberay_operator`                | KubeRay 활성화                                      | `false`                  |
| `huggingface_token`                      | 환경에서 사용할 Hugging Face 토큰                   | `DUMMY_TOKEN_REPLACE_ME` |
| `enable_rayserve_ha_elastic_cache_redis` | ElastiCache를 사용한 Rayserve 고가용성 활성화       | `false`                  |
| `enable_torchx_etcd`                     | torchx용 etcd 활성화                                | `false`                  |
| `enable_mpi_operator`                    | MPI Operator 활성화                                 | `false`                  |
| `enable_aibrix_stack`                    | AIBrix 스택 활성화                                  | `false`                  |
| `aibrix_stack_version`                   | AIBrix 스택 버전                                    | `v0.2.1`                 |
| `enable_aws_load_balancer_controller`    | AWS Load Balancer Controller 활성화                 | `true`                   |
| `enable_service_mutator_webhook`         | AWS Load Balancer Controller용 service-mutator 웹훅 활성화 | `false`            |
| `enable_ingress_nginx`                   | ingress-nginx 애드온 활성화                         | `true`                   |
| `enable_cert_manager`                    | Cert Manager 활성화                                 | `false`                  |
| `enable_slurm_operator`                  | Slinky Slurm Operator 활성화 (Cert Manager 포함)    | `false`                  |

### JupyterHub

| 변수 이름                       | 설명                                                                                  | 기본값  |
|-------------------------------|---------------------------------------------------------------------------------------|---------|
| `jupyter_hub_auth_mechanism`  | JupyterHub에 사용할 인증 메커니즘 [`dummy` \| `cognito` \| `oauth`]                   | `dummy` |
| `cognito_custom_domain`       | Hosted UI 인증 엔드포인트를 위한 Cognito 도메인 접두사                                | `eks`   |
| `acm_certificate_domain`      | ACM 인증서에 사용되는 도메인 이름                                                     | `""`    |
| `jupyterhub_domain`           | JupyterHub 도메인 이름 (cognito 또는 oauth 사용 시에만)                               | `""`    |
| `oauth_jupyter_client_id`     | JupyterHub용 OAuth 클라이언트 ID. OAuth 사용 시에만 필요                              | `""`    |
| `oauth_jupyter_client_secret` | OAuth 클라이언트 시크릿. OAuth 사용 시에만 필요                                       | `""`    |
| `oauth_username_key`          | 사용자 이름을 위한 OAuth 필드 (예: `preferred_username`). OAuth 사용 시에만 필요       | `""`    |

## 커스텀 스택

위의 변수를 사용하면 자신의 필요에 맞는 새로운 환경을 쉽게 구성할 수 있습니다. `infra` 폴더에 간단한 `blueprint.tfvars`가 포함된 `custom` 폴더가 있습니다. 위의 변수를 적절한 값으로 추가하면 원하는 애드온이 배포된 환경을 생성하여 기호에 맞게 커스터마이징할 수 있습니다. 변수를 추가한 후 `infra/custom` 루트에 있는 `install.sh`를 실행하세요.
