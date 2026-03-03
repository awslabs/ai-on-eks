# 문제 해결

AI on Amazon EKS(AIoEKS) 설치 문제에 대한 문제 해결 정보를 찾을 수 있습니다

## 오류: local-exec provisioner error

local-exec provisioner 실행 중 다음 오류가 발생하는 경우:

```sh
Error: local-exec provisioner error \
with module.eks-blueprints.module.emr_on_eks["data_team_b"].null_resource.update_trust_policy,\
 on .terraform/modules/eks-blueprints/modules/emr-on-eks/main.tf line 105, in resource "null_resource" \
 "update_trust_policy":│ 105: provisioner "local-exec" {│ │ Error running command 'set -e│ │ aws emr-containers update-role-trust-policy \
 │ --cluster-name emr-on-eks \│ --namespace emr-data-team-b \│ --role-name emr-on-eks-emr-eks-data-team-b
```
### 문제 설명:
오류 메시지는 사용 중인 AWS CLI 버전에 emr-containers 명령이 없음을 나타냅니다. 이 문제는 AWS CLI 버전 2.0.54에서 해결되었습니다.

### 해결 방법
문제를 해결하려면 다음 명령을 실행하여 AWS CLI 버전을 2.0.54 이상으로 업데이트하세요:

```sh
pip install --upgrade awscliv2
```

AWS CLI 버전을 업데이트하면 프로비저닝 프로세스 중에 필요한 emr-containers 명령을 사용할 수 있으며 성공적으로 실행할 수 있습니다.

문제가 계속되거나 추가 지원이 필요한 경우 자세한 내용은 [AWS CLI GitHub 이슈](https://github.com/aws/aws-cli/issues/6162)를 참조하거나 지원 팀에 문의하여 추가 안내를 받으세요.

## Terraform Destroy 중 타임아웃

### 문제 설명:
고객은 환경을 삭제하는 동안 특히 VPC가 삭제될 때 타임아웃을 경험할 수 있습니다. 이는 vpc-cni 구성 요소와 관련된 알려진 문제입니다.

### 증상:

환경이 삭제된 후에도 ENI(Elastic Network Interface)가 서브넷에 연결된 상태로 유지됩니다.
ENI와 연결된 EKS 관리형 보안 그룹이 EKS에 의해 삭제될 수 없습니다.
### 해결 방법:
이 문제를 해결하려면 아래 권장 해결 방법을 따르세요:

제공된 `cleanup.sh` 스크립트를 사용하여 리소스를 올바르게 정리합니다. 블루프린트에 포함된 `cleanup.sh` 스크립트를 실행합니다.
이 스크립트는 남아있는 ENI 및 관련 보안 그룹의 제거를 처리합니다.


## 오류: could not download chart
차트를 다운로드하려고 할 때 다음 오류가 발생하는 경우:

```sh
│ Error: could not download chart: failed to download "oci://public.ecr.aws/karpenter/karpenter" at version "v0.18.1"
│
│   with module.eks_blueprints_kubernetes_addons.module.karpenter[0].module.helm_addon.helm_release.addon[0],
│   on .terraform/modules/eks_blueprints_kubernetes_addons/modules/kubernetes-addons/helm-addon/main.tf line 1, in resource "helm_release" "addon":
│    1: resource "helm_release" "addon" {
│
```

문제를 해결하려면 아래 단계를 따르세요:

### 문제 설명:
오류 메시지는 지정된 차트를 다운로드하는 데 실패했음을 나타냅니다. 이 문제는 Karpenter 설치 중 Terraform의 버그로 인해 발생할 수 있습니다.

### 해결 방법:
문제를 해결하려면 다음 단계를 시도할 수 있습니다:

ECR로 인증: 차트가 있는 ECR(Elastic Container Registry)로 인증하기 위해 다음 명령을 실행합니다:

```sh
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws
```
terraform apply 재실행: Terraform 구성을 다시 적용하기 위해 --auto-approve 플래그와 함께 terraform apply 명령을 다시 실행합니다:
```sh
terraform apply --auto-approve
```

ECR로 인증하고 terraform apply 명령을 다시 실행하면 설치 프로세스 중에 필요한 차트를 성공적으로 다운로드할 수 있습니다.

## EKS 클러스터와 인증하기 위한 Terraform apply/destroy 오류
```
ERROR:
╷
│ Error: Get "http://localhost/api/v1/namespaces/kube-system/configmaps/aws-auth": dial tcp [::1]:80: connect: connection refused
│
│   with module.eks.kubernetes_config_map_v1_data.aws_auth[0],
│   on .terraform/modules/eks/main.tf line 550, in resource "kubernetes_config_map_v1_data" "aws_auth":
│  550: resource "kubernetes_config_map_v1_data" "aws_auth" {
│
╵
```

**해결 방법:**
이 상황에서 Terraform은 데이터 리소스를 새로 고치고 EKS 클러스터와 인증할 수 없습니다.
[여기](https://github.com/terraform-aws-modules/terraform-aws-eks/issues/1234)에서 논의를 참조하세요

먼저 exec 플러그인을 사용하여 이 접근 방식을 시도하세요.

```terraform
provider "kubernetes" {
  host                   = module.eks_blueprints.eks_cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks_blueprints.eks_cluster_certificate_authority_data)

  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args = ["eks", "get-token", "--cluster-name", module.eks_blueprints.eks_cluster_id]
  }
}


```

위의 변경 후에도 문제가 지속되면 로컬 kube config 파일을 사용하는 대체 접근 방식을 사용할 수 있습니다.
참고: 이 접근 방식은 프로덕션에 이상적이지 않을 수 있습니다. 로컬 kube config로 클러스터를 적용/삭제하는 데 도움이 됩니다.

1. 클러스터용 로컬 kubeconfig 생성

```bash
aws eks update-kubeconfig --name <EKS_CLUSTER_NAME> --region <CLUSTER_REGION>
```

2. config_path만 사용하여 아래 구성으로 `providers.tf` 파일을 업데이트합니다.

```terraform
provider "kubernetes" {
    config_path = "<HOME_PATH>/.kube/config"
}

provider "helm" {
    kubernetes {
        config_path = "<HOME_PATH>/.kube/config"
    }
}

provider "kubectl" {
    config_path = "<HOME_PATH>/.kube/config"
}
```

## EMR Containers Virtual Cluster (dhwtlq9yx34duzq5q3akjac00) delete: unexpected state 'ARRESTED'

"waiting for EMR Containers Virtual Cluster (xwbc22787q6g1wscfawttzzgb) delete: unexpected state 'ARRESTED', wanted target ''. last error: %!s(nil)"라는 오류 메시지가 나타나면 아래 단계를 따라 문제를 해결할 수 있습니다:

참고: `<REGION>`을 가상 클러스터가 위치한 적절한 AWS 리전으로 바꾸세요.

1. 터미널 또는 명령 프롬프트를 엽니다.
2. 다음 명령을 실행하여 "ARRESTED" 상태의 가상 클러스터를 나열합니다:

```sh
aws emr-containers list-virtual-clusters --region <REGION> --states ARRESTED \
--query 'virtualClusters[0].id' --output text
```
이 명령은 "ARRESTED" 상태의 가상 클러스터 ID를 검색합니다.

3. 다음 명령을 실행하여 가상 클러스터를 삭제합니다:

```sh
aws emr-containers list-virtual-clusters --region <REGION> --states ARRESTED \
--query 'virtualClusters[0].id' --output text | xargs -I{} aws emr-containers delete-virtual-cluster \
--region <REGION> --id {}
```
`<VIRTUAL_CLUSTER_ID>`를 이전 단계에서 얻은 가상 클러스터 ID로 바꾸세요.

이 명령을 실행하면 "ARRESTED" 상태의 가상 클러스터를 삭제할 수 있습니다. 이렇게 하면 예기치 않은 상태 문제가 해결되고 추가 작업을 진행할 수 있습니다.

## 네임스페이스 종료 문제

네임스페이스가 "Terminating" 상태에서 멈추고 삭제할 수 없는 문제가 발생하면 다음 명령을 사용하여 네임스페이스의 finalizer를 제거할 수 있습니다:

참고: `<namespace>`를 삭제하려는 네임스페이스 이름으로 바꾸세요.

```sh
NAMESPACE=<namespace>
kubectl get namespace $NAMESPACE -o json | sed 's/"kubernetes"//' | kubectl replace --raw "/api/v1/namespaces/$NAMESPACE/finalize" -f -
```

이 명령은 네임스페이스 세부 정보를 JSON 형식으로 검색하고, "kubernetes" finalizer를 제거하고, 네임스페이스에서 finalizer를 제거하기 위해 replace 작업을 수행합니다. 이렇게 하면 네임스페이스가 종료 프로세스를 완료하고 성공적으로 삭제될 수 있습니다.

이 작업을 수행하는 데 필요한 권한이 있는지 확인하세요. 문제가 계속되거나 추가 지원이 필요한 경우 지원 팀에 연락하여 추가 안내 및 문제 해결 단계를 받으세요.

## KMS Alias AlreadyExistsException

Terraform 설치 또는 재배포 중에 다음과 같은 오류가 발생할 수 있습니다: `AlreadyExistsException: An alias with the name ...` already exists. 이는 생성하려는 KMS 별칭이 AWS 계정에 이미 존재할 때 발생합니다.

```
│ Error: creating KMS Alias (alias/eks/trainium-inferentia): AlreadyExistsException: An alias with the name arn:aws:kms:us-west-2:23423434:alias/eks/trainium-inferentia already exists
│
│   with module.eks.module.kms.aws_kms_alias.this["cluster"],
│   on .terraform/modules/eks.kms/main.tf line 452, in resource "aws_kms_alias" "this":
│  452: resource "aws_kms_alias" "this" {
│
```

**해결 방법:**

이 문제를 해결하려면 aws kms delete-alias 명령을 사용하여 기존 KMS 별칭을 삭제합니다. 명령을 실행하기 전에 별칭 이름과 리전을 업데이트하세요.


```sh
aws kms delete-alias --alias-name <KMS_ALIAS_NAME> --region <ENTER_REGION>
```

## 오류: creating CloudWatch Logs Log Group

Terraform이 AWS 계정에 이미 존재하기 때문에 CloudWatch Logs 로그 그룹을 생성할 수 없습니다.

```
╷
│ Error: creating CloudWatch Logs Log Group (/aws/eks/trainium-inferentia/cluster): operation error CloudWatch Logs: CreateLogGroup, https response error StatusCode: 400, RequestID: 5c34c47a-72c6-44b2-a345-925824f24d38, ResourceAlreadyExistsException: The specified log group already exists
│
│   with module.eks.aws_cloudwatch_log_group.this[0],
│   on .terraform/modules/eks/main.tf line 106, in resource "aws_cloudwatch_log_group" "this":
│  106: resource "aws_cloudwatch_log_group" "this" {

```

**해결 방법:**

로그 그룹 이름과 리전을 업데이트하여 기존 로그 그룹을 삭제합니다.

```sh
aws logs delete-log-group --log-group-name <LOG_GROUP_NAME> --region <ENTER_REGION>
```

## Karpenter 오류 - Service Linked Role 누락

Karpenter가 새 인스턴스를 생성하려고 할 때 아래 오류가 발생합니다.

```
"error":"launching nodeclaim, creating instance, with fleet error(s), AuthFailure.ServiceLinkedRoleCreationNotPermitted: The provided credentials do not have permission to create the service-linked role for EC2 Spot Instances."}
```

**해결 방법:**

`ServiceLinkedRoleCreationNotPermitted` 오류를 피하려면 사용 중인 AWS 계정에 서비스 연결 역할을 생성해야 합니다.

```sh
aws iam create-service-linked-role --aws-service-name spot.amazonaws.com
```


## 배포 후 Karpenter가 CrashLoopBackOff 상태 - STS 오류

배포 후 karpenter 파드를 확인할 때 다음 로그와 함께 CrashLoopBackOff 상태입니다:
```
{"level":"ERROR","time":"2025-09-30T12:52:27.746Z","logger":"controller","message":"ec2 api connectivity check failed","commit":"13242ea","error":"operation error EC2: DescribeInstanceTypes,
get identity: get credentials: failed to refresh cached credentials, failed to retrieve credentials, operation error STS: AssumeRoleWithWebIdentity, https response error StatusCode: 403, RequestID: xxx, RegionDisabledException: STS is not activated in this region for account:xxxxx. Your account administrator can activate STS in this region using the IAM Console."}
```

## 해결 방법:

"STS is not activated in this region for account" 메시지는 AWS 계정 관리자가 요청이 이루어지는 특정 리전에서 임시 자격 증명을 생성하기 위한 요청이 성공하기 전에 AWS Security Token Service(STS)를 활성화해야 함을 의미합니다. 계정 설정 페이지의 AWS IAM 콘솔을 사용하여 리전에 대해 STS를 활성화할 수 있으며, 여기에서 현재 활성화된 리전도 표시됩니다.

문서 링크: https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_temp_enable-regions.html

## 오류: AmazonEKS_CNI_IPv6_Policy does not exist
IPv6를 지원하는 솔루션을 배포할 때 다음 오류가 발생하는 경우:

```sh
│ Error: attaching IAM Policy (arn:aws:iam::1234567890:policy/AmazonEKS_CNI_IPv6_Policy) to IAM Role (core-node-group-eks-node-group-20241111182906854800000003): operation error IAM: AttachRolePolicy, https response error StatusCode: 404, RequestID: 9c99395a-ce3d-4a05-b119-538470a3a9f7, NoSuchEntity: Policy arn:aws:iam::1234567890:policy/AmazonEKS_CNI_IPv6_Policy does not exist or is not attachable.
```

### 문제 설명:
Amazon VPC CNI 플러그인은 IPv6 주소를 할당하기 위해 IAM 권한이 필요하므로 IAM 정책을 생성하고 CNI가 사용할 역할과 연결해야 합니다. 그러나 각 IAM 정책 이름은 동일한 AWS 계정에서 고유해야 합니다. 정책이 terraform 스택의 일부로 생성되고 여러 번 배포되면 충돌이 발생합니다.

이 오류를 해결하려면 아래 명령으로 정책을 생성해야 합니다. AWS 계정당 한 번만 수행하면 됩니다.

### 해결 방법:

1. 다음 텍스트를 복사하여 vpc-cni-ipv6-policy.json이라는 파일에 저장합니다.

```sh
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ec2:AssignIpv6Addresses",
                "ec2:DescribeInstances",
                "ec2:DescribeTags",
                "ec2:DescribeNetworkInterfaces",
                "ec2:DescribeInstanceTypes"
            ],
            "Resource": ""
        },
        {
            "Effect": "Allow",
            "Action": [
                "ec2:CreateTags"
            ],
            "Resource": [
                "arn:aws:ec2::*:network-interface/*"
            ]
        }
    ]
}
```

2. IAM 정책을 생성합니다.

```sh
aws iam create-policy --policy-name AmazonEKS_CNI_IPv6_Policy --policy-document file://vpc-cni-ipv6-policy.json
```

3. 블루프린트의 `install.sh` 스크립트를 다시 실행합니다
