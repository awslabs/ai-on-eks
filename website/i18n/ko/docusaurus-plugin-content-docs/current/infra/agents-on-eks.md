---
sidebar_label: EKS 기반 에이전트
---

# EKS 기반 에이전트

Agents on EKS 인프라는 오픈소스 도구를 사용하여 AI 에이전트를 지속적으로 빌드, 배포 및 평가할 수 있는 안전하고 확장 가능하며 신뢰할 수 있는 환경을 배포합니다.

## 왜 필요한가?

대규모 AI 에이전트를 구축하고 운영하려면 추론 인프라 이상의 것이 필요합니다. 에이전트에는 다음이 필요합니다:

- **소스 제어 및 CI/CD**: 에이전트 코드 및 구성의 버전 관리
- **관측성**: 에이전트 동작 추적, 성능 평가 및 문제 디버깅
- **영구 메모리**: 임베딩 저장 및 검색 증강 생성(RAG) 활성화
- **도구 오케스트레이션**: MCP(Model Context Protocol) 서버 관리 및 검색

이 인프라는 이러한 구성 요소를 응집력 있는 플랫폼으로 통합하여 팀이 신뢰성을 유지하면서 에이전트 개발을 빠르게 반복할 수 있도록 합니다.

## 사용 사례

- **에이전트 개발**: 통합된 소스 제어 및 CI/CD 파이프라인으로 AI 에이전트를 빌드하고 테스트
- **에이전트 평가**: Langfuse를 사용하여 에이전트 실행을 추적하고, 출력을 평가하고, 시간에 따른 성능을 추적
- **RAG 애플리케이션**: Milvus를 사용하여 지식 증강 에이전트를 위한 임베딩 저장 및 검색
- **MCP 도구 관리**: 게이트웨이 레지스트리를 통해 MCP 서버 검색 및 관리
- **멀티 에이전트 시스템**: 공유 인프라로 여러 에이전트를 배포하고 오케스트레이션

## 아키텍처

이 인프라는 다음을 생성합니다:

- **Amazon VPC**: 여러 가용 영역에 걸친 퍼블릭 및 프라이빗 서브넷
- **Amazon EKS 클러스터**: 중요한 애드온을 위한 관리형 노드 그룹 포함
- **Karpenter**: 워크로드 요구에 따른 지능형 노드 오토스케일링
- **GitLab**: 소스 제어, 컨테이너 레지스트리 및 CI/CD 파이프라인
- **Langfuse**: 에이전트 관측성, 추적 및 평가
- **Milvus**: 벡터 저장 및 유사성 검색
- **MCP Gateway Registry**: 도구 검색 및 관리

### 핵심 구성 요소

| 구성 요소 | 목적 |
|-----------|------|
| [GitLab](https://about.gitlab.com/) | 에이전트 코드를 위한 소스 제어, 컨테이너 레지스트리 및 CI/CD |
| [Langfuse](https://langfuse.com/) | LLM 관측성, 추적, 프롬프트 관리 및 평가 |
| [Milvus](https://milvus.io/) | 임베딩 및 유사성 검색을 위한 벡터 데이터베이스 |
| [MCP Gateway Registry](https://github.com/agentic-community/mcp-gateway-registry) | Model Context Protocol 서버의 검색 및 관리 |
| [Karpenter](https://karpenter.sh/) | Kubernetes 노드 오토스케일링 |
| [ArgoCD](https://argo-cd.readthedocs.io/) | GitOps 지속적 배포 |

## 사전 요구 사항

### 도메인 및 인증서 설정

GitLab은 유효한 TLS 인증서가 필요하며, 이를 위해서는 도메인을 소유해야 합니다. 기존 도메인의 서브도메인을 사용할 수 있습니다.

1. **Route53 호스팅 영역 생성**

   [AWS 문서를 참조하여 호스팅 영역을 생성](https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/CreatingHostedZone.html)하세요. 서브도메인의 경우 `subdomain.domain.tld` 패턴을 따라 이름을 지정합니다.

2. **(선택 사항) 서브도메인으로 구성**

   서브도메인을 사용하는 경우, 기본 도메인에 호스팅 영역을 [서브도메인](https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/CreatingNewSubdomain.html)으로 추가합니다.

3. **ACM 인증서 생성**

   [ACM 문서를 참조하여 도메인에 대한 인증서를 생성](https://docs.aws.amazon.com/acm/latest/userguide/gs-acm-request-public.html)하세요.

### 필요한 도구

- 적절한 권한으로 구성된 AWS CLI
- Terraform >= 1.0
- kubectl
- Helm >= 3.0

## 배포

### 1단계: 복제 및 이동

```bash
git clone https://github.com/awslabs/ai-on-eks.git
cd ai-on-eks/infra/solutions/agents-on-eks
```

### 2단계: 변수 구성

도메인을 설정하려면 `terraform/blueprint.tfvars`를 편집합니다:

```hcl
name                        = "aioeks-agents"
enable_langfuse             = true
enable_gitlab               = true
enable_external_dns         = true
enable_milvus               = true
enable_mcp_gateway_registry = true
max_user_namespaces         = 16384
acm_certificate_domain      = "agents.example.com"  # 도메인으로 업데이트
allowed_inbound_cidrs       = "0.0.0.0/0"           # 인바운드 IP 제한
```

### 3단계: 배포

```bash
./install.sh
```

배포는 약 20분이 소요됩니다.

### 4단계: kubectl 구성

배포 후 클러스터에 액세스하도록 kubectl을 구성합니다:

```bash
aws eks update-kubeconfig --name aioeks-agents --region us-west-2
```

## 서비스 액세스

### GitLab

GitLab은 `https://gitlab.<your-domain>`에서 사용할 수 있습니다. root 비밀번호를 검색합니다:

```bash
kubectl get secret gitlab-gitlab-initial-root-password -n gitlab -o jsonpath='{.data.password}' | base64 -d
```

### Langfuse

포트 포워딩을 통해 Langfuse에 액세스합니다:

```bash
kubectl port-forward svc/langfuse 3000:3000 -n langfuse
```

그런 다음 브라우저에서 `http://localhost:3000`을 엽니다.

### Milvus

클러스터 내에서 `milvus.milvus.svc.cluster.local:19530`으로 Milvus에 연결합니다.

### MCP Gateway Registry

MCP Gateway Registry는 `https://mcpregistry.<your-domain>`에서 사용할 수 있습니다.

## 구성 옵션

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `name` | 클러스터 이름 | `aioeks-agents` |
| `region` | AWS 리전 | `us-west-2` |
| `eks_cluster_version` | EKS 버전 | `1.34` |
| `acm_certificate_domain` | TLS 인증서용 도메인 | `""` (필수) |
| `allowed_inbound_cidrs` | 로드 밸런서를 통해 허용되는 CIDR 범위 | `0.0.0.0/0` |
| `enable_langfuse` | Langfuse 배포 | `true` |
| `enable_gitlab` | GitLab 배포 | `true` |
| `enable_milvus` | Milvus 배포 | `true` |
| `enable_mcp_gateway_registry` | MCP Gateway Registry 배포 | `true` |
| `enable_external_dns` | Route53용 External DNS 활성화 | `true` |

### 인바운드 액세스 제한

`allowed_inbound_cidrs` 변수는 로드 밸런서를 통해 서비스에 액세스할 수 있는 IP 범위를 제어합니다. 조직의 IP 범위로 제한하세요:

```hcl
allowed_inbound_cidrs = "10.0.0.0/8,192.168.1.0/24"
```

CI/CD 파이프라인을 위해 CIDR에 개발자 IP와 GitLab Runner 노드 IP가 포함되어 있는지 확인하세요.

## 정리

인프라를 삭제하려면:

```bash
cd terraform/_LOCAL
./cleanup.sh
```

## 다음 단계

- 에이전트 코드를 위한 GitLab CI/CD 파이프라인 구성
- 추적을 위한 Langfuse 프로젝트 및 API 키 설정
- 임베딩 저장을 위한 Milvus 컬렉션 생성
- 게이트웨이 레지스트리에 MCP 서버 등록
