---
sidebar_label: EKS 기반 JupyterHub
---
import CollapsibleContent from '@site/src/components/CollapsibleContent';

:::warning
EKS에서 ML 모델을 배포하려면 GPU 또는 Neuron 인스턴스에 대한 액세스가 필요합니다. 배포가 작동하지 않는 경우 이러한 리소스에 대한 액세스가 누락되어 있는 경우가 많습니다. 또한 일부 배포 패턴은 Karpenter 자동 확장 및 정적 노드 그룹에 의존합니다. 노드가 초기화되지 않는 경우 Karpenter 또는 노드 그룹의 로그를 확인하여 문제를 해결하세요.
:::

# EKS 기반 JupyterHub

[JupyterHub](https://jupyter.org/hub)는 사용자가 Jupyter 노트북 및 기타 Jupyter 호환 환경에 액세스하고 상호 작용할 수 있게 해주는 강력한 다중 사용자 서버입니다. 여러 사용자가 동시에 노트북에 액세스하고 활용할 수 있는 협업 플랫폼을 제공하여 사용자 간의 협업과 공유를 촉진합니다. JupyterHub를 통해 사용자는 자신만의 격리된 컴퓨팅 환경("스포너"라고 함)을 만들고 해당 환경 내에서 Jupyter 노트북 또는 기타 대화형 컴퓨팅 환경을 시작할 수 있습니다. 이를 통해 각 사용자에게 파일, 코드 및 컴퓨팅 리소스를 포함한 자체 작업 공간이 제공됩니다.

## EKS 기반 JupyterHub
Amazon Elastic Kubernetes Service(EKS)에 JupyterHub를 배포하면 JupyterHub의 다양성과 Kubernetes의 확장성 및 유연성이 결합됩니다. 이 블루프린트를 통해 사용자는 JupyterHub 프로필의 도움으로 EKS에서 다중 테넌트 JupyterHub 플랫폼을 구축할 수 있습니다. 각 사용자를 위한 EFS 공유 파일 시스템을 활용하여 노트북 공유를 쉽게 하고 개별 EFS 스토리지를 제공하여 사용자 파드가 삭제되거나 만료되더라도 데이터를 안전하게 저장할 수 있습니다. 사용자가 로그인하면 기존 EFS 볼륨 아래의 모든 스크립트와 데이터에 액세스할 수 있습니다.

EKS의 기능을 활용하면 사용자의 요구에 맞게 JupyterHub 환경을 원활하게 확장하여 효율적인 리소스 활용과 최적의 성능을 보장할 수 있습니다. EKS를 사용하면 자동 확장, 고가용성, 업데이트 및 업그레이드의 쉬운 배포와 같은 Kubernetes 기능을 활용할 수 있습니다. 이를 통해 사용자에게 신뢰할 수 있고 강력한 JupyterHub 경험을 제공하여 효과적으로 협업, 탐색 및 데이터 분석을 수행할 수 있도록 지원합니다.

EKS에서 JupyterHub를 시작하려면 이 가이드의 지침에 따라 JupyterHub 환경을 설정하고 구성하세요.

<CollapsibleContent header={<h3><span>솔루션 배포</span></h3>}>

이 [블루프린트](https://github.com/awslabs/ai-on-eks/tree/main/infra/jupyterhub)는 다음 구성 요소를 배포합니다:

- 2개의 프라이빗 서브넷과 2개의 퍼블릭 서브넷을 포함한 새 샘플 VPC를 생성합니다. VPC 문서 링크
- 퍼블릭 서브넷용 인터넷 게이트웨이와 프라이빗 서브넷용 NAT 게이트웨이를 설정합니다.
- 퍼블릭 엔드포인트(데모 목적으로만)와 코어 관리형 노드 그룹이 있는 EKS 클러스터 컨트롤 플레인을 생성합니다.
- JupyterHub를 설정하기 위해 [JupyterHub Helm 차트](https://hub.jupyter.org/helm-chart/)를 배포합니다.
- 개인 스토리지용 하나와 공유 스토리지용 하나, 두 개의 EFS 스토리지 마운트를 설정합니다.
- 선택 사항: [Amazon Cognito](https://aws.amazon.com/cognito/) 사용자 풀을 사용하여 사용자를 인증합니다. Cognito 문서 링크

이 블루프린트를 따르면 다양한 AWS 서비스를 활용하여 사용자를 위한 협업적이고 확장 가능한 플랫폼을 제공하는 EKS에서 JupyterHub 환경을 쉽게 배포하고 구성할 수 있습니다.

<CollapsibleContent header={<h3><span>사전 요구 사항</span></h3>}>

**유형 1: 도메인 이름과 로드 밸런서 없이 JupyterHub 배포**:

이 접근 방식은 JupyterHub에서 포트 포워딩(`kubectl port-forward svc/proxy-public 8080:80 -n jupyterhub`)을 사용합니다. 개발 및 테스트 환경에서 테스트하는 데 유용합니다. 프로덕션 배포의 경우 Cognito와 같은 적절한 인증 메커니즘으로 JupyterHub를 호스팅하기 위한 사용자 지정 도메인 이름이 필요합니다. 프로덕션에서의 인증에는 접근 방식 2를 사용하세요.

**유형 2: 사용자 지정 도메인 이름, ACM 및 NLB로 JupyterHub 배포**:

이 접근 방식은 도메인 이름을 만들고 ACM 인증서를 얻어야 합니다. 도메인 이름과 인증서를 위해 조직 또는 플랫폼 팀과 협력해야 합니다. 자체 인증 메커니즘 또는 AWS Cognito를 사용할 수 있습니다.

머신에 다음 도구가 설치되어 있는지 확인하세요.

1. [aws cli](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
2. [kubectl](https://Kubernetes.io/docs/tasks/tools/)
3. [terraform](https://learn.hashicorp.com/tutorials/terraform/install-cli)

:::info

아래는 유형 2 배포에만 필요하며 사용자 지정 도메인 이름과 ACM 인증서가 필요합니다

:::

4. **도메인 이름**: 사용자 지정 도메인으로 JupyterHub WebUI를 호스팅하려면 자체 도메인 이름을 가져와야 합니다. 테스트 목적으로 [ChangeIP](https://www.changeip.com/accounts/index.php)와 같은 무료 도메인 서비스 제공업체를 사용하여 테스트 도메인을 만들 수 있습니다. 그러나 ChangeIP 또는 유사한 서비스를 사용하여 JupyterHub로 프로덕션 또는 개발 클러스터를 호스팅하는 것은 권장되지 않습니다. 이러한 서비스 사용에 대한 이용 약관을 검토하세요.
5. **SSL 인증서**: 도메인에 첨부할 신뢰할 수 있는 인증 기관(CA) 또는 웹 호스팅 제공업체를 통해 SSL 인증서를 얻어야 합니다. 테스트 환경의 경우 OpenSSL 서비스를 사용하여 자체 서명 인증서를 생성할 수 있습니다.

```bash
openssl req -newkey rsa:2048 -nodes -keyout key.pem -x509 -days 365 -out certificate.pem
```

인증서를 만들 때 와일드카드를 사용하면 단일 인증서로 도메인과 모든 하위 도메인을 보호할 수 있습니다.
서비스는 개인 키와 자체 서명 인증서를 생성합니다.
인증서를 생성하기 위한 샘플 프롬프트:

![](../img/Cert_Install.png)


6. AWS Certificate Manager로 인증서 가져오기

텍스트 편집기에서 개인 키(`key.pem`)를 열고 내용을 ACM의 개인 키 섹션에 복사합니다. 마찬가지로 `certificate.pem` 파일의 내용을 인증서 본문 섹션에 복사하고 제출합니다.

   ![](../img/ACM.png)

   ACM에서 콘솔에서 인증서가 올바르게 설치되었는지 확인합니다.

   ![](../img/Cert_List.png)

</CollapsibleContent>

**JupyterHub 인증 옵션**

이 블루프린트는 `dummy`, `cognito`, `oauth`의 세 가지 인증 메커니즘을 지원합니다. 이 게시물에서는 쉬운 시연을 위해 dummy 메커니즘을 사용하며 프로덕션에 권장되는 인증 메커니즘이 아닙니다. 프로덕션 준비 설정에는 Authenticators 페이지에서 찾을 수 있는 cognito 방법 또는 기타 지원되는 인증 메커니즘을 사용하는 것을 강력히 권장합니다.

<a id="배포"></a>
### 배포

**유형 1 배포 구성 변경:**

`blueprint.tfvars` 파일에서 `region` 변수만 업데이트합니다.

**유형 2 배포 구성 변경:**

다음 변수로 `blueprint.tfvars` 파일을 업데이트합니다:
 - `acm_certificate_domain`
 - `jupyterhub_domain`
 - `jupyter_hub_auth_mechanism=cognito`

**유형 3 배포 구성 변경:**

다음 변수로 `blueprint.tfvars` 파일을 업데이트합니다:
- `acm_certificate_domain`
- `jupyterhub_domain`
- `jupyter_hub_auth_mechanism=oauth`
- `oauth_domain`
- `oauth_jupyter_client_id`
- `oauth_jupyter_client_secret`
- `oauth_username_key`

리포지토리 복제

```bash
git clone https://github.com/awslabs/ai-on-eks.git
```

블루프린트 디렉터리 중 하나로 이동합니다

```bash
cd ai-on-eks/infra/jupyterhub && chmod +x install.sh
```

:::info

배포가 완료되지 않으면 install.sh를 다시 실행하세요

:::


</CollapsibleContent>


<CollapsibleContent header={<h3><span>리소스 확인</span></h3>}>

먼저 새로 생성된 Amazon EKS 클러스터에 연결하도록 kubeconfig를 구성해야 합니다. 다음 명령을 사용하고 필요한 경우 `us-west-2`를 특정 AWS 리전으로 바꾸세요:

```bash
aws eks --region us-west-2 update-kubeconfig --name jupyterhub-on-eks
```
이제 다음을 실행하여 다양한 네임스페이스에서 파드의 상태를 확인할 수 있습니다. 주요 배포에 주의를 기울이세요:

```bash
kubectl get pods -A
```

이 확인 단계는 모든 필요한 구성 요소가 올바르게 작동하는지 확인하는 데 필수적입니다. 모든 것이 정상이면 Amazon EKS의 JupyterHub 환경이 데이터 및 머신 러닝 팀을 지원할 준비가 되었다는 것을 확신하고 진행할 수 있습니다.

JupyterHub 애드온이 실행 중인지 확인하려면 컨트롤러 및 웹훅에 대한 애드온 배포가 RUNNING 상태인지 확인하세요.

다음 명령을 실행합니다

```bash
kubectl get pods -n jupyterhub
```

이 블루프린트에서 배포한 Karpenter 프로비저너를 확인합니다. JupyterHub 프로필에서 특정 노드를 스핀업하기 위해 이러한 프로비저너가 어떻게 사용되는지 설명하겠습니다.

```bash
kubectl get provisioners
```

이 블루프린트에서 생성한 Persistent Volume Claims(PVC)를 확인합니다. 각각 고유한 목적을 제공합니다. efs-persist라는 Amazon EFS 볼륨은 각 JupyterHub 단일 사용자 파드의 개별 홈 디렉터리로 마운트되어 각 사용자에게 전용 공간을 보장합니다. 반면 efs-persist-shared는 모든 JupyterHub 단일 사용자 파드에 마운트되는 특수 PVC로 사용자 간의 협업 노트북 공유를 용이하게 합니다. 이와 함께 JupyterHub, Kube Prometheus Stack 및 KubeCost 배포를 강력하게 지원하기 위해 추가 Amazon EBS 볼륨이 프로비저닝되었습니다.

```bash
kubectl get pvc -A
```

</CollapsibleContent>

### 유형 1 배포: JupyterHub 로그인

**포트 포워드로 JupyterHub 노출**:

웹 사용자 인터페이스를 로컬에서 보기 위해 JupyterHub 서비스에 액세스할 수 있도록 아래 명령을 실행합니다. 현재 dummy 배포는 `ClusterIP`가 있는 Web UI 서비스만 설정한다는 점에 유의하세요. 이를 내부 또는 인터넷 연결 로드 밸런서로 사용자 지정하려면 JupyterHub Helm 차트 값 파일에서 필요한 조정을 할 수 있습니다.

```bash
kubectl port-forward svc/proxy-public 8080:80 -n jupyterhub
```

**로그인:** 웹 브라우저에서 [http://localhost:8080/](http://localhost:8080/)로 이동합니다. 사용자 이름으로 `user-1`을 입력하고 아무 비밀번호나 선택합니다.
![alt text](../img/image.png)

서버 옵션 선택: 로그인하면 선택할 수 있는 다양한 노트북 인스턴스 프로필이 표시됩니다. `Data Engineering (CPU)` 서버는 전통적인 CPU 기반 노트북 작업용입니다. `Elyra` 서버는 파이프라인을 빠르게 개발할 수 있는 [Elyra](https://github.com/elyra-ai/elyra) 기능을 제공합니다: ![workflow](../img/elyra-workflow.png). `Trainium` 및 `Inferentia` 서버는 노트북 서버를 Trainium 및 Inferentia 노드에 배포하여 가속화된 워크로드를 허용합니다. `Time Slicing` 및 `MIG`는 GPU 공유를 위한 두 가지 다른 전략입니다. 마지막으로 `Data Science (GPU)` 서버는 NVIDIA GPU에서 실행되는 전통적인 서버입니다.

이 타임슬라이싱 기능 시연을 위해 **Data Science (GPU + Time-Slicing – G5)** 프로필을 사용합니다. 이 옵션을 선택하고 Start 버튼을 선택하세요.

![alt text](../img/notebook-server-list.png)

`g5.2xlarge` 인스턴스 유형으로 Karpenter가 생성한 새 노드는 [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin)에서 제공하는 타임슬라이싱 기능을 활용하도록 구성되었습니다. 이 기능을 사용하면 단일 GPU를 여러 할당 가능한 단위로 분할하여 효율적인 GPU 활용이 가능합니다. 이 경우 NVIDIA device plugin Helm 차트 구성 맵에서 `4`개의 할당 가능한 GPU를 정의했습니다. 아래는 노드 상태입니다:

GPU: 노드는 NVIDIA device plugin의 타임슬라이싱 기능을 통해 4개의 GPU로 구성됩니다. 이를 통해 노드가 다양한 워크로드에 GPU 리소스를 더 유연하게 할당할 수 있습니다.

```yaml
status:
  capacity:
    cpu: '8'                           # 노드에 8개의 CPU가 있습니다
    ephemeral-storage: 439107072Ki     # 노드의 총 임시 스토리지 용량은 439107072 KiB입니다
    hugepages-1Gi: '0'                 # 노드에 0개의 1Gi hugepage가 있습니다
    hugepages-2Mi: '0'                 # 노드에 0개의 2Mi hugepage가 있습니다
    memory: 32499160Ki                 # 노드의 총 메모리 용량은 32499160 KiB입니다
    nvidia.com/gpu: '4'                # 노드에 타임슬라이싱을 통해 구성된 총 4개의 GPU가 있습니다
    pods: '58'                         # 노드는 최대 58개의 파드를 수용할 수 있습니다
  allocatable:
    cpu: 7910m                         # 7910 밀리코어의 CPU가 할당 가능합니다
    ephemeral-storage: '403607335062'  # 403607335062 KiB의 임시 스토리지가 할당 가능합니다
    hugepages-1Gi: '0'                 # 0개의 1Gi hugepage가 할당 가능합니다
    hugepages-2Mi: '0'                 # 0개의 2Mi hugepage가 할당 가능합니다
    memory: 31482328Ki                 # 31482328 KiB의 메모리가 할당 가능합니다
    nvidia.com/gpu: '4'                # 4개의 GPU가 할당 가능합니다
    pods: '58'                         # 58개의 파드가 할당 가능합니다

```

**두 번째 사용자(`user-2`) 환경 설정**:

GPU 타임슬라이싱이 작동하는 것을 시연하기 위해 다른 Jupyter Notebook 인스턴스를 프로비저닝합니다. 이번에는 두 번째 사용자의 파드가 이전에 설정한 GPU 타임슬라이싱 구성을 활용하여 첫 번째 사용자와 동일한 노드에 예약되었는지 확인합니다. 이를 달성하려면 아래 단계를 따르세요:

시크릿 브라우저 창에서 JupyterHub 열기: 새 **시크릿 창**의 웹 브라우저에서 http://localhost:8080/로 이동합니다. 사용자 이름으로 `user-2`를 입력하고 아무 비밀번호나 선택합니다.

서버 옵션 선택: 로그인 후 서버 옵션 페이지가 표시됩니다. **Data Science (GPU + Time-Slicing – G5)** 라디오 버튼을 선택하고 Start를 선택합니다.

![alt text](../img/image-2.png)

파드 배치 확인: 이 파드 배치는 `user-1`과 달리 몇 초밖에 걸리지 않습니다. Kubernetes 스케줄러가 `user-1` 파드에서 생성한 기존 `g5.2xlarge` 노드에 파드를 배치할 수 있기 때문입니다. `user-2`도 동일한 도커 이미지를 사용하므로 도커 이미지를 가져오는 데 지연이 없고 로컬 캐시를 활용했습니다.

터미널을 열고 다음 명령을 실행하여 새 Jupyter Notebook 파드가 어디에 예약되었는지 확인합니다:

```bash
kubectl get pods -n jupyterhub -owide | grep -i user
```

`user-1`과 `user-2` 파드가 모두 동일한 노드에서 실행되고 있는지 확인합니다. 이는 **GPU 타임슬라이싱** 구성이 예상대로 작동하고 있음을 확인합니다.

:::info

자세한 내용은 [AWS 블로그: Building multi-tenant JupyterHub Platforms on Amazon EKS](https://aws.amazon.com/blogs/containers/building-multi-tenant-jupyterhub-platforms-on-amazon-eks/)를 확인하세요

:::

### 유형 2 배포(선택 사항): Amazon Cognito를 통해 JupyterHub 로그인

로드 밸런서 DNS 이름으로 JupyterHub 도메인에 대한 `CNAME` DNS 레코드를 ChangeIP에 추가합니다.

![](../img/CNAME.png)

:::info
ChangeIP의 CNAME 값 필드에 로드 밸런서 DNS 이름을 추가할 때 로드 밸런서 DNS 이름 끝에 점(`.`)을 추가해야 합니다.
:::

이제 브라우저에서 도메인 URL을 입력하면 JupyterHub 로그인 페이지로 리디렉션됩니다.

![](../img/Cognito-Sign-in.png)


Cognito 가입 및 로그인 프로세스를 따라 로그인합니다.

![](../img/Cognito-Sign-up.png)

성공적인 로그인은 로그인한 사용자를 위한 JupyterHub 환경을 엽니다.

![](../img/jupyter_launcher.png)

JupyterHub에서 공유 및 개인 디렉터리 설정을 테스트하려면 다음 단계를 따르세요:
1. 런처 대시보드에서 터미널 창을 엽니다.

![](../img/jupyter_env.png)

2.  명령을 실행합니다

```bash
df -h
```
생성된 EFS 마운트를 확인합니다. 각 사용자의 개인 홈 디렉터리는 `/home/jovyan`에서 사용할 수 있습니다. 공유 디렉터리는 `/home/shared`에서 사용할 수 있습니다

### 유형 3 배포(선택 사항): OAuth(Keycloak)를 통해 JupyterHub 로그인

참고: OAuth 제공업체에 따라 약간 다르게 보일 수 있습니다.

로드 밸런서 DNS 이름으로 JupyterHub 도메인에 대한 `CNAME` DNS 레코드를 ChangeIP에 추가합니다.

![](../img/CNAME.png)

:::info
ChangeIP의 CNAME 값 필드에 로드 밸런서 DNS 이름을 추가할 때 로드 밸런서 DNS 이름 끝에 점(`.`)을 추가해야 합니다.
:::

이제 브라우저에서 도메인 URL을 입력하면 JupyterHub 로그인 페이지로 리디렉션됩니다.

![](../img/oauth.png)

Keycloak 가입 및 로그인 프로세스를 따라 로그인합니다.

![](../img/keycloak-login.png)

성공적인 로그인은 로그인한 사용자를 위한 JupyterHub 환경을 엽니다.

![](../img/jupyter_launcher.png)


<CollapsibleContent header={<h3><span>정리</span></h3>}>

:::caution
AWS 계정에 원치 않는 요금이 청구되지 않도록 이 배포 중에 생성된 모든 AWS 리소스를 삭제하세요.
:::

이 스크립트는 -target 옵션을 사용하여 모든 리소스가 올바른 순서로 삭제되도록 환경을 정리합니다.

```bash
cd ai-on-eks/infra/jupyterhub/ && chmod +x cleanup.sh
./cleanup.sh
```

</CollapsibleContent>
