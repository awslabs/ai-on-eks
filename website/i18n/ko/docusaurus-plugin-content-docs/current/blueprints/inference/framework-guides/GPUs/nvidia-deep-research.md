---
title: EKS의 NVIDIA Enterprise RAG 및 AI-Q Research Assistant
sidebar_position: 9
---

import CollapsibleContent from '@site/src/components/CollapsibleContent';

:::warning
EKS에 Enterprise RAG 및 AI-Q를 배포하려면 GPU 인스턴스(g5, p4 또는 p5 제품군)에 대한 액세스가 필요합니다. 이 블루프린트는 동적 GPU 프로비저닝을 위해 [Karpenter](https://karpenter.sh/) 오토스케일링에 의존합니다.
:::

:::info
이 블루프린트는 두 가지 배포 옵션을 제공합니다: **Enterprise RAG Blueprint** (NVIDIA Nemotron 및 NeMo Retriever 모델을 사용한 멀티모달 문서 처리) 또는 전체 **AI-Q Research Assistant** (웹 검색을 통한 자동화된 연구 보고서 추가). 둘 다 동적 GPU 오토스케일링과 함께 Amazon EKS에서 실행됩니다.

출처: [NVIDIA RAG Blueprint](https://github.com/NVIDIA-AI-Blueprints/rag) | [NVIDIA AI-Q Research Assistant](https://github.com/NVIDIA-AI-Blueprints/aiq-research-assistant)
:::

# Amazon EKS의 NVIDIA Enterprise RAG 및 AI-Q Research Assistant

## NVIDIA AI-Q Research Assistant란?

[NVIDIA AI-Q Research Assistant](https://build.nvidia.com/nvidia/aiq)는 어디서나 작동할 수 있고, 자체 데이터 소스로 정보를 제공받으며, 몇 시간 분량의 연구를 몇 분 만에 종합할 수 있는 맞춤형 AI 연구원을 생성하는 AI 기반 연구 어시스턴트입니다. AI-Q NVIDIA Blueprint를 통해 개발자는 AI 에이전트를 엔터프라이즈 데이터에 연결하고 추론 및 도구를 사용하여 효율성과 정밀도로 심층적인 소스 자료를 추출할 수 있습니다.

### 주요 기능

**고급 연구 자동화:**
- 빠른 보고서 합성을 위한 **5배 빠른 토큰 생성**
- 더 나은 의미론적 정확도로 **15배 빠른 데이터 수집**
- 효율성과 정밀도로 다양한 데이터 세트 요약
- 자동으로 포괄적인 연구 보고서 생성

**NVIDIA NeMo Agent Toolkit:**
- 에이전트 워크플로우 개발 및 최적화 용이
- 다양한 프레임워크에 걸쳐 워크플로우 통합, 평가, 감사 및 디버그
- 최적화 기회 식별
- 각 작업에 가장 적합한 에이전트와 도구를 유연하게 선택하고 연결

**NVIDIA NeMo Retriever를 통한 고급 의미론적 쿼리:**
- 멀티모달 PDF 데이터 추출 및 검색 (텍스트, 표, 차트, 인포그래픽)
- 15배 빠른 엔터프라이즈 데이터 수집
- 3배 낮은 검색 지연 시간
- 다국어 및 교차 언어 지원
- 정확도 향상을 위한 리랭킹
- GPU 가속 인덱스 생성 및 검색

**Llama Nemotron을 통한 빠른 추론:**
- 최고의 정확도와 최저 지연 시간 추론 기능
- [Llama-3.3-Nemotron-Super-49B-v1.5](https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1_5) 추론 모델 사용
- 데이터 소스 분석 및 패턴 식별
- 포괄적인 연구를 기반으로 솔루션 제안
- 엔터프라이즈 데이터로 지원되는 컨텍스트 인식 생성

**웹 검색 통합:**
- Tavily API로 구동되는 실시간 웹 검색
- 현재 정보로 온프레미스 소스 보완
- 내부 문서를 넘어 연구 확장

### AI-Q 구성 요소

[공식 AI-Q 아키텍처](https://github.com/NVIDIA-AI-Blueprints/aiq-research-assistant)에 따르면:

**1. NVIDIA AI Workbench**
- 에이전트 워크플로우를 위한 간소화된 개발 환경
- 로컬 테스트 및 사용자 정의
- 다양한 LLM의 손쉬운 구성
- NVIDIA NeMo Agent Toolkit 통합

**2. NVIDIA RAG Blueprint**
- 대규모 온프레미스 멀티모달 문서 세트 쿼리를 위한 솔루션
- 텍스트, 이미지, 표 및 차트 추출 지원
- GPU 가속을 통한 의미론적 검색 및 검색
- AI-Q의 연구 기능을 위한 기반

**3. NVIDIA NeMo Retriever Microservices**
- 멀티모달 문서 수집
- 그래픽 요소 감지
- 표 구조 추출
- 텍스트 인식을 위한 PaddleOCR
- 15배 빠른 데이터 수집

**4. NVIDIA NIM Microservices**
- LLM 및 비전 모델을 위한 최적화된 추론 컨테이너
- [Llama-3.3-Nemotron-Super-49B-v1.5](https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1_5) 추론 모델
- 보고서 생성을 위한 Llama-3.3-70B-Instruct 모델
- GPU 가속 추론

**5. 웹 검색 (Tavily)**
- 실시간 웹 검색으로 온프레미스 소스 보완
- 내부 문서를 넘어 연구 확장
- 웹 보강 연구 보고서 지원

## NVIDIA Enterprise RAG Blueprint란?

[NVIDIA Enterprise RAG Blueprint](https://build.nvidia.com/nvidia/build-an-enterprise-rag-pipeline)는 검색과 생성 모두를 위한 확장 가능하고 사용자 정의 가능한 파이프라인을 구축하기 위한 완전한 기반을 제공하는 프로덕션 준비 참조 워크플로우입니다. NVIDIA NeMo Retriever 모델과 NVIDIA Llama Nemotron 모델로 구동되는 이 블루프린트는 높은 정확도, 강력한 추론 및 엔터프라이즈 규모의 처리량에 최적화되어 있습니다.

멀티모달 데이터 수집, 고급 검색, 리랭킹 및 반영 기술에 대한 내장 지원과 LLM 기반 워크플로우와의 원활한 통합을 통해 수백만 개의 문서에서 텍스트, 표, 차트, 오디오 및 인포그래픽에 걸쳐 언어 모델을 엔터프라이즈 데이터에 연결하여 진정한 컨텍스트 인식 및 생성적 응답을 가능하게 합니다.

### 주요 기능

**데이터 수집 및 처리:**
- 텍스트, 표, 차트 및 인포그래픽이 포함된 **멀티모달 PDF 데이터 추출**
- **오디오 파일 수집** 지원
- 사용자 정의 메타데이터 지원
- 문서 요약
- 엔터프라이즈 규모로 수백만 개의 문서 지원

**벡터 데이터베이스 및 검색:**
- 문서 세트에 걸친 다중 컬렉션 검색 가능
- 밀집 및 희소 검색을 통한 **하이브리드 검색**
- 정확도 향상을 위한 리랭킹
- GPU 가속 인덱스 생성 및 검색
- **플러그 가능 벡터 데이터베이스** 아키텍처:
  - ElasticSearch 지원
  - Milvus 지원
  - OpenSearch Serverless 지원 (이 배포에서 사용)
- 복잡한 쿼리를 위한 쿼리 분해
- 동적 메타데이터 필터 생성

**멀티모달 및 고급 생성:**
- 답변 생성에서 선택적 **Vision Language Model (VLM)** 지원
- VLM을 통한 옵트인 이미지 캡션
- 대화형 Q&A를 위한 다중 턴 대화
- 동시 사용자를 위한 다중 세션 지원
- 선택적 반영으로 정확도 향상

**거버넌스 및 안전:**
- 선택적 프로그래밍 가능 가드레일로 콘텐츠 안전 개선
- 엔터프라이즈급 보안 기능
- 데이터 프라이버시 및 규정 준수 제어

**관측성 및 텔레메트리:**
- 평가 스크립트 포함 (RAGAS 프레임워크)
- 분산 추적을 위한 OpenTelemetry 지원
- 추적 시각화를 위한 Zipkin 통합
- 메트릭 및 모니터링을 위한 Grafana 대시보드
- 성능 프로파일링 및 최적화 도구

**개발자 기능:**
- 테스트 및 데모용 사용자 인터페이스 포함
- DRA를 사용한 GPU 공유를 위한 NIM Operator 지원
- 네이티브 Python 라이브러리 지원
- 쉬운 통합을 위한 OpenAI 호환 API
- 분해 가능하고 사용자 정의 가능한 아키텍처
- 기능 확장을 위한 플러그인 시스템

### Enterprise RAG 사용 사례

Enterprise RAG Blueprint는 독립적으로 또는 대규모 시스템의 구성 요소로 사용할 수 있습니다:

- 문서 저장소 전반의 **엔터프라이즈 검색**
- 조직 지식 베이스용 **지식 어시스턴트**
- 도메인별 애플리케이션용 **생성형 코파일럿**
- 특정 산업에 맞춤화된 **수직 AI 워크플로우**
- 에이전트 워크플로우의 **기반 구성 요소** (AI-Q Research Assistant처럼)
- 컨텍스트 인식 응답을 통한 **고객 지원 자동화**
- 대규모 **문서 분석** 및 요약

엔터프라이즈 검색, 지식 어시스턴트, 생성형 코파일럿 또는 수직 AI 워크플로우를 구축하든, RAG용 NVIDIA AI Blueprint는 프로토타입에서 프로덕션으로 자신 있게 이동하는 데 필요한 모든 것을 제공합니다. 독립적으로 사용하거나, 다른 NVIDIA Blueprint와 결합하거나, 더 고급 추론 기반 애플리케이션을 지원하기 위해 에이전트 워크플로우에 통합할 수 있습니다.

## 개요

이 블루프린트는 **[NVIDIA AI-Q Research Assistant](https://github.com/NVIDIA-AI-Blueprints/aiq-research-assistant)**를 Amazon EKS에 구현하며, 포괄적인 연구 기능을 위해 [NVIDIA RAG Blueprint](https://github.com/NVIDIA-AI-Blueprints/rag)와 AI-Q 구성 요소를 결합합니다.

### 배포 옵션

이 블루프린트는 사용 사례에 따라 두 가지 배포 모드를 지원합니다:

**옵션 1: Enterprise RAG Blueprint**
- 멀티모달 문서 처리와 함께 NVIDIA Enterprise RAG Blueprint 배포
- NeMo Retriever 마이크로서비스 및 OpenSearch 통합 포함
- 적합 용도: 사용자 정의 RAG 애플리케이션, 문서 Q&A 시스템, 지식 베이스 구축

**옵션 2: 전체 AI-Q Research Assistant**
- 옵션 1의 모든 것에 AI-Q 구성 요소 추가
- Tavily API를 통한 웹 검색 기능으로 자동화된 연구 보고서 생성 추가
- 적합 용도: 포괄적인 연구 작업, 자동화된 보고서 생성, 웹 보강 연구

두 배포 모두 [Karpenter](https://karpenter.sh/) 오토스케일링과 엔터프라이즈 보안 기능을 포함합니다. 옵션 1로 시작하여 필요에 따라 나중에 AI-Q 구성 요소를 추가할 수 있습니다.

### 배포 접근 방식

**이 설정 프로세스의 이유는?**
이 구현은 여러 단계를 포함하지만 여러 가지 이점을 제공합니다:

- **완전한 인프라**: VPC, EKS 클러스터, OpenSearch Serverless 및 모니터링 스택을 자동으로 프로비저닝
- **엔터프라이즈 기능**: 보안, 모니터링 및 확장성 기능 포함
- **AWS 통합**: [Karpenter](https://karpenter.sh/) 오토스케일링, EKS Pod Identity 인증 및 관리형 AWS 서비스 활용
- **재현 가능**: Infrastructure as Code로 환경 전반에 걸쳐 일관된 배포 보장

### 주요 기능

**성능 최적화:**
- **[Karpenter](https://karpenter.sh/) 오토스케일링**: 워크로드 요구에 따른 동적 GPU 노드 프로비저닝
- **지능형 인스턴스 선택**: 최적의 GPU 인스턴스 유형(G5, P4, P5) 자동 선택
- **빈 패킹**: 여러 워크로드에 걸친 효율적인 GPU 활용

**엔터프라이즈 준비:**
- **OpenSearch Serverless**: 자동 확장을 통한 관리형 벡터 데이터베이스
- **Pod Identity 인증**: Pod에서 안전한 AWS IAM 액세스를 위한 EKS Pod Identity
- **관측성 스택**: GPU 모니터링을 위한 Prometheus, Grafana 및 DCGM
- **보안 액세스**: 제어된 서비스 액세스를 위한 Kubernetes 포트 포워딩

## 아키텍처

### AI-Q Research Assistant 아키텍처

배포는 [Karpenter](https://karpenter.sh/) 기반 동적 프로비저닝과 함께 Amazon EKS를 사용합니다:

![NVIDIA AI-Q on EKS](../../img/nvidia-deep-research-arch.png)


### Enterprise RAG Blueprint 아키텍처

![RAG Pipeline with OpenSearch](../../img/nvidia-rag-opensearch-arch.png)

[RAG 파이프라인](https://github.com/NVIDIA-AI-Blueprints/rag)은 여러 특수 NIM 마이크로서비스를 통해 문서를 처리합니다:

**1. Llama-3.3-Nemotron-Super-49B-v1.5**
- [고급 추론 모델](https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1_5)
- RAG 및 보고서 작성 모두를 위한 기본 추론 및 생성
- 쿼리 재작성 및 분해
- 필터 표현식 생성

**2. 임베딩 및 리랭킹**
- LLama 3.2 NV-EmbedQA: 2048차원 임베딩
- LLama 3.2 NV-RerankQA: 관련성 점수 매기기

**3. NV-Ingest 파이프라인**
- **PaddleOCR**: 이미지에서 텍스트 추출
- **Page Elements**: 문서 레이아웃 이해
- **Graphic Elements**: 차트 및 다이어그램 감지
- **Table Structure**: 표 형식 데이터 추출

**4. AI-Q Research Assistant 구성 요소**
- 보고서 생성을 위한 Llama-3.3-70B-Instruct 모델 (선택 사항, 2 GPU)
- Tavily API를 통한 웹 검색
- 연구 워크플로우를 위한 백엔드 오케스트레이션

## 사전 요구 사항

:::info 중요 - 비용 정보
이 배포는 상당한 비용이 발생할 수 있는 GPU 인스턴스를 사용합니다. 자세한 비용 추정은 이 가이드 끝의 [비용 고려 사항](#비용-고려-사항)을 참조하십시오. **사용하지 않을 때는 항상 리소스를 정리하십시오.**
:::

**시스템 요구 사항**: AWS CLI 액세스가 있는 모든 Linux/macOS 시스템

다음 도구를 설치하십시오:

- **AWS CLI**: 적절한 권한으로 구성됨 ([설치 가이드](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html))
- **kubectl**: Kubernetes 명령줄 도구 ([설치 가이드](https://kubernetes.io/docs/tasks/tools/install-kubectl/))
- **helm**: Kubernetes 패키지 관리자 ([설치 가이드](https://helm.sh/docs/intro/install/))
- **terraform**: Infrastructure as code 도구 ([설치 가이드](https://learn.hashicorp.com/tutorials/terraform/install-cli))
- **git**: 버전 제어 ([설치 가이드](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git))

### 필수 API 토큰

- **NGC API 토큰**: NVIDIA NIM 컨테이너 및 AI Foundation 모델에 액세스하는 데 필요
  - **먼저 다음 옵션 중 하나를 통해 가입하십시오** (API 키는 이러한 계정 중 하나가 있어야만 작동합니다):
    - **옵션 1 - NVIDIA Developer Program** (빠른 시작):
      - [여기](https://build.nvidia.com/)에서 가입
      - POC 및 개발 워크로드용 무료 계정
      - 테스트 및 평가에 이상적
    - **옵션 2 - NVIDIA AI Enterprise** (프로덕션):
      - [AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-ozgjkov6vq3l6)를 통해 구독
      - 전체 지원 및 SLA가 포함된 엔터프라이즈 라이선스
      - 프로덕션 배포에 필요
  - **그런 다음 API 키를 생성하십시오**:
    - 옵션 1 또는 2를 통해 가입한 후 [NGC Personal Keys](https://org.ngc.nvidia.com/setup/personal-keys)에서 API 키를 생성합니다
    - 이 키를 잘 보관하십시오 - 배포 시 필요합니다
- **[Tavily API 키](https://tavily.com/)**: **AI-Q Research Assistant에 선택 사항**
  - AI-Q에서 웹 검색 기능 활성화
  - AI-Q는 이 키 없이도 RAG 전용 모드로 작동 가능
  - Enterprise RAG 전용 배포에는 필요 없음
  - [Tavily](https://tavily.com/)에서 계정 생성
  - 대시보드에서 API 키 생성
  - 이 키를 잘 보관하십시오 - AI-Q에서 웹 검색을 원하면 배포 시 필요합니다

### GPU 인스턴스 액세스

AWS 계정이 GPU 인스턴스에 액세스할 수 있는지 확인하십시오. 이 블루프린트는 [Karpenter](https://karpenter.sh/) NodePool을 통해 여러 인스턴스 제품군을 지원합니다:

**지원되는 GPU 인스턴스 제품군:**

| 인스턴스 제품군 | GPU 유형 | 성능 프로파일 | 사용 사례 |
|----------------|----------|---------------------|----------|
| **G5** (기본) | NVIDIA A10G | 비용 효율적, 24GB VRAM | 일반 워크로드, 개발 |
| **G6e** | NVIDIA L40S | 균형 잡힌, 48GB VRAM | 고메모리 모델 |
| **P4d/P4de** | NVIDIA A100 | 고성능, 40/80GB VRAM | 대규모 배포 |
| **P5/P5e/P5en** | NVIDIA H100 | 초고성능, 80GB VRAM | 최대 성능 |

> **참고**: G5 인스턴스는 접근 가능한 시작점을 제공하기 위해 Helm 값에 미리 구성되어 있습니다. Helm 값 파일의 `nodeSelector`를 편집하여 P4/P5/G6e 인스턴스로 전환할 수 있습니다 - 인프라 변경은 필요 없습니다.

<CollapsibleContent header={<h4><span>GPU 인스턴스 유형 사용자 정의 (선택 사항)</span></h4>}>

:::tip GPU 인스턴스 유연성
이 블루프린트는 비용 효율적인 시작점을 제공하기 위해 **G5 인스턴스 (A10G GPU)**로 미리 구성되어 있습니다. 그러나 Helm 값 파일을 수정하여 **P4 (A100) 또는 P5 (H100) 인스턴스로 쉽게 전환**할 수 있습니다. 인프라에는 G5, G6, G6e, P4 및 P5 인스턴스 제품군을 위한 Karpenter NodePool이 포함되어 있습니다 - 성능 및 예산 요구 사항에 맞게 `nodeSelector` 레이블을 변경하기만 하면 됩니다.
:::

모든 구성 요소는 자동 프로비저닝을 위해 Karpenter 레이블을 사용합니다. **기본 구성 (G5 인스턴스)**:

```yaml
# 예: 8-GPU 워크로드 (49B/70B 모델)
nodeSelector:
  karpenter.k8s.aws/instance-family: g5  # G5 (A10G GPU) 사용
  karpenter.k8s.aws/instance-size: 48xlarge  # 8x A10G
  karpenter.sh/capacity-type: on-demand

# 예: 1-GPU 워크로드 (임베딩, 리랭킹, OCR)
nodeSelector:
  karpenter.k8s.aws/instance-family: g5  # G5 (A10G GPU) 사용
  karpenter.k8s.aws/instance-size: 12xlarge  # 최대 4x A10G
```

**다른 GPU 유형을 사용하려면** Helm 값에서 `instance-family`를 업데이트하십시오:

```yaml
# P5 (H100 GPU)의 경우 - 최고 성능
nodeSelector:
  karpenter.k8s.aws/instance-family: p5
  karpenter.k8s.aws/instance-size: 48xlarge  # 8x H100

# P4 (A100 GPU)의 경우 - 고성능
nodeSelector:
  karpenter.k8s.aws/instance-family: p4d
  karpenter.k8s.aws/instance-size: 24xlarge  # 8x A100

# G6e (L40S GPU)의 경우 - 균형 잡힌 성능
nodeSelector:
  karpenter.k8s.aws/instance-family: g6e
  karpenter.k8s.aws/instance-size: 48xlarge  # 8x L40S
```

**수동 노드 생성 불필요** - Karpenter가 `nodeSelector` 구성에 따라 적절한 인스턴스를 자동으로 프로비저닝합니다!

</CollapsibleContent>

## 시작하기

시작하려면 저장소를 클론하십시오:

```bash
git clone https://github.com/awslabs/ai-on-eks.git
cd ai-on-eks
```

## 배포

이 블루프린트는 두 가지 배포 방법을 제공합니다:

<CollapsibleContent header={<h2><span>옵션 A: 자동화된 배포 (권장)</span></h2>}>

제공된 bash 스크립트를 사용하여 전체 배포 프로세스를 자동화합니다.

> **팁**: 전체 구성 제어가 포함된 자세한 수동 배포 단계는 아래 [옵션 B: 수동 배포](#옵션-b-수동-배포)를 참조하십시오.

<a id="1단계-인프라-배포"></a>
### 1단계: 인프라 배포

인프라 디렉토리로 이동하고 설치 스크립트를 실행합니다:

```bash
cd infra/nvidia-deep-research
./install.sh
```

이것은 완전한 환경을 프로비저닝합니다:
- **VPC**: 서브넷, 보안 그룹, NAT 게이트웨이
- **EKS 클러스터**: 동적 GPU 프로비저닝을 위한 [Karpenter](https://karpenter.sh/) 포함
- **OpenSearch Serverless**: Pod Identity 인증을 통한 벡터 데이터베이스
- **모니터링 스택**: Prometheus, Grafana 및 AI/ML 관측성
- **[Karpenter](https://karpenter.sh/) NodePool**: G5, G6, G6e, P4, P5 인스턴스 지원

**소요 시간**: 15-20분

> **인프라 준비 완료**: Terraform이 성공적으로 완료되면 인프라가 배포되어 준비됩니다.

<a id="2단계-환경-설정"></a>
### 2단계: 환경 설정

설정 스크립트를 실행하여 환경을 구성합니다:

```bash
./deploy.sh setup
```

이 스크립트는:
- EKS 클러스터에 액세스하도록 kubectl 구성
- NGC 및 Tavily API 키 수집
- 클러스터 준비 상태 확인 (Karpenter, NodePool, OpenSearch)
- GPU 노드를 위한 Karpenter 제한 패치
- `.env` 파일에 구성 저장

<a id="3단계-opensearch-이미지-빌드"></a>
### 3단계: OpenSearch 이미지 빌드

RAG 소스를 클론하고 OpenSearch를 통합하고 사용자 정의 Docker 이미지를 빌드합니다:

```bash
./deploy.sh build
```

**대기 시간**: 이미지 빌드에 10-15분

<a id="4단계-애플리케이션-배포"></a>
### 4단계: 애플리케이션 배포

사용 사례에 따라 선택하십시오:

#### 1) Enterprise RAG만 배포

AI-Q 연구 기능 없이 문서 Q&A용:

```bash
./deploy.sh rag
```

**대기 시간**: 15-25분

**배포되는 구성 요소:**
- **49B Nemotron 모델** (8 GPU) - [Karpenter](https://karpenter.sh/)가 g5.48xlarge를 프로비저닝
- **임베딩 및 리랭킹 모델** (각각 1 GPU)
- **데이터 수집 모델** (각각 1 GPU)
- **RAG 서버** (OpenSearch Serverless 통합)
- **프론트엔드** (사용자 상호 작용용)

---

#### 2) AI-Q Research Assistant 배포

AI-Q는 Enterprise RAG Blueprint와 선택적 웹 검색 기능이 포함된 자동화된 연구 보고서 생성을 포함합니다.

##### 옵션 A: 한 번에 모두 배포 (권장 - 더 빠름)

RAG와 AI-Q를 병렬로 배포합니다:

```bash
./deploy.sh all
```

**대기 시간**: 25-30분

**배포되는 모든 구성 요소:**
- **RAG**: 49B Nemotron 모델, 임베딩 및 리랭킹 모델, 데이터 수집 모델, RAG 서버, 프론트엔드
- **AI-Q**: 70B Instruct 모델, AIRA 백엔드, 프론트엔드, 웹 검색 (Tavily API 키 제공 시)

##### 옵션 B: 순차적 배포

먼저 RAG를 배포한 다음 AI-Q를 추가합니다:

```bash
# 1단계: RAG 배포
./deploy.sh rag

# 2단계: AI-Q 배포
# AI-Q는 웹 검색 없이도 작동 가능 (Tavily API는 선택 사항)
./deploy.sh aira
```

**대기 시간**: RAG에 15-25분, 그 다음 AI-Q에 20-30분 (총 35-55분)


---

</CollapsibleContent>

<CollapsibleContent header={<h2><span>옵션 B: 수동 배포</span></h2>}>

각 구성 요소와 구성을 이해하기 위한 자세한 수동 단계를 따르십시오. 학습, 사용자 정의 또는 문제 해결에 이상적입니다.

<a id="1단계-인프라-배포"></a>
### 1단계: 인프라 배포

인프라 디렉토리로 이동합니다:

```bash
cd infra/nvidia-deep-research
```

설치 스크립트를 실행합니다:

```bash
./install.sh
```

**소요 시간**: 15-20분

프로비저닝되는 것:
- 퍼블릭 및 프라이빗 서브넷이 있는 VPC
- [Karpenter](https://karpenter.sh/)가 포함된 EKS 클러스터
- OpenSearch Serverless 컬렉션
- 모니터링 스택 (Prometheus, Grafana)
- GPU 인스턴스용 [Karpenter](https://karpenter.sh/) NodePool

> **인프라 준비 완료**: Terraform이 성공적으로 완료되면 인프라가 배포되어 준비됩니다.

<a id="2단계-환경-설정"></a>
### 2단계: 환경 설정

kubectl을 구성하고 필요한 환경 변수를 설정합니다:

```bash
# 클러스터 구성
export CLUSTER_NAME="nvidia-deep-research"
export REGION="us-west-2"

# kubectl 구성
aws eks update-kubeconfig --region $REGION --name $CLUSTER_NAME

# 클러스터 연결 확인
kubectl get nodes

# AWS 계정 ID 가져오기
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# OpenSearch 구성
export OPENSEARCH_SERVICE_ACCOUNT="opensearch-access-sa"
export OPENSEARCH_NAMESPACE="rag"
export COLLECTION_NAME="osv-vector-dev"

# Terraform 출력에서 OpenSearch 엔드포인트 가져오기
export OPENSEARCH_ENDPOINT=$(cd terraform/_LOCAL && terraform output -raw opensearch_collection_endpoint)

echo "OpenSearch Endpoint: $OPENSEARCH_ENDPOINT"

# NGC API 키 (필수)
export NGC_API_KEY="<YOUR_NGC_API_KEY>"

# AI-Q용 Tavily API 키 (선택 사항 - 웹 검색 활성화)
export TAVILY_API_KEY="<YOUR_TAVILY_API_KEY>"  # RAG만 배포하거나 웹 검색 없는 AI-Q의 경우 생략
```

<a id="3단계-karpenter-nodepool-제한-구성"></a>
### 3단계: [Karpenter](https://karpenter.sh/) NodePool 제한 구성

G5 GPU NodePool의 메모리 제한을 늘립니다:

```bash
kubectl patch nodepool g5-gpu-karpenter --type='json' -p='[{"op": "replace", "path": "/spec/limits/memory", "value": "2000Gi"}]'
```

이를 통해 [Karpenter](https://karpenter.sh/)가 모든 모델에 충분한 GPU 노드를 프로비저닝할 수 있습니다 (1000Gi에서 2000Gi로).

<a id="4단계-opensearch-통합-및-docker-이미지-빌드"></a>
### 4단계: OpenSearch 통합 및 Docker 이미지 빌드

RAG 소스 코드를 클론하고 OpenSearch 구현을 추가합니다:

```bash
# RAG 소스 코드 클론
git clone -b v2.3.0 https://github.com/NVIDIA-AI-Blueprints/rag.git rag

# OpenSearch 구현 다운로드
COMMIT_HASH="47cd8b345e5049d49d8beb406372de84bd005abe"
curl -L https://github.com/NVIDIA/nim-deploy/archive/${COMMIT_HASH}.tar.gz | tar xz --strip=5 nim-deploy-${COMMIT_HASH}/cloud-service-providers/aws/blueprints/deep-research-blueprint-eks/opensearch

# RAG 소스에 OpenSearch 구현 복사
cp -r opensearch/vdb/opensearch rag/src/nvidia_rag/utils/vdb/
cp opensearch/main.py rag/src/nvidia_rag/ingestor_server/main.py
cp opensearch/vdb/__init__.py rag/src/nvidia_rag/utils/vdb/__init__.py
cp opensearch/pyproject.toml rag/pyproject.toml

# NGC 레지스트리 로그인
docker login nvcr.io  # 사용자 이름: $oauthtoken, 비밀번호: NGC API 키

# ECR 로그인
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# ECR에 OpenSearch 지원 RAG 이미지 빌드 및 푸시
./opensearch/build-opensearch-images.sh
```

**대기 시간**: 이미지 빌드에 10-15분

<a id="5단계-enterprise-rag-blueprint-배포"></a>
### 5단계: Enterprise RAG Blueprint 배포

OpenSearch 지원 이미지를 사용하여 RAG Blueprint를 배포합니다:

```bash
# 배포 변수 설정
export ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
export IMAGE_TAG="2.3.0-opensearch"

# OpenSearch 구성으로 RAG 배포
helm upgrade --install rag -n rag \
  https://helm.ngc.nvidia.com/nvidia/blueprint/charts/nvidia-blueprint-rag-v2.3.0.tgz \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --create-namespace \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  --set serviceAccount.create=false \
  --set serviceAccount.name=$OPENSEARCH_SERVICE_ACCOUNT \
  --set image.repository="${ECR_REGISTRY}/nvidia-rag-server" \
  --set image.tag="${IMAGE_TAG}" \
  --set ingestor-server.image.repository="${ECR_REGISTRY}/nvidia-rag-ingestor" \
  --set ingestor-server.image.tag="${IMAGE_TAG}" \
  --set envVars.APP_VECTORSTORE_URL="${OPENSEARCH_ENDPOINT}" \
  --set envVars.APP_VECTORSTORE_AWS_REGION="${REGION}" \
  --set ingestor-server.envVars.APP_VECTORSTORE_URL="${OPENSEARCH_ENDPOINT}" \
  --set ingestor-server.envVars.APP_VECTORSTORE_AWS_REGION="${REGION}" \
  -f helm/rag-values-os.yaml

# OpenSearch 서비스 계정을 사용하도록 ingestor-server 패치
kubectl patch deployment ingestor-server -n rag \
  -p "{\"spec\":{\"template\":{\"spec\":{\"serviceAccountName\":\"$OPENSEARCH_SERVICE_ACCOUNT\"}}}}"
```

**대기 시간**: 모델 다운로드 및 GPU 프로비저닝에 10-20분

RAG 배포 확인:

```bash
# RAG 네임스페이스의 모든 Pod 확인
kubectl get all -n rag

# 모든 Pod가 준비될 때까지 대기
kubectl wait --for=condition=ready pod -l app.kubernetes.io/instance=rag -n rag --timeout=600s

# 서비스 계정 확인
kubectl get pod -n rag -l app.kubernetes.io/component=rag-server -o jsonpath='{.items[0].spec.serviceAccountName}'
kubectl get pod -n rag -l app=ingestor-server -o jsonpath='{.items[0].spec.serviceAccountName}'
```

GPU 메트릭용 DCGM ServiceMonitor 배포:

```bash
# RAG의 Prometheus를 인프라 DCGM Exporter에 연결하는 ServiceMonitor 배포
kubectl apply -f - <<EOF
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: dcgm-exporter
  namespace: rag
  labels:
    release: rag
spec:
  namespaceSelector:
    matchNames:
      - monitoring
  selector:
    matchLabels:
      app.kubernetes.io/name: dcgm-exporter
  endpoints:
    - port: metrics
      interval: 15s
      path: /metrics
EOF
```

이 ServiceMonitor를 통해 `rag` 네임스페이스의 Prometheus 인스턴스가 `monitoring` 네임스페이스에서 실행 중인 DCGM Exporter의 GPU 메트릭을 검색하고 스크래핑할 수 있습니다.

**NVIDIA DCGM Grafana 대시보드 배포 (선택 사항이지만 권장):**

```bash
# 공식 NVIDIA DCGM 대시보드 다운로드 및 배포 (데이터소스 수정 포함)
curl -s https://grafana.com/api/dashboards/12239 | jq -r '.json' | \
    jq 'walk(if type == "object" and has("datasource") and (.datasource | type == "string") then .datasource = {"type": "prometheus", "uid": "prometheus"} else . end)' \
    > /tmp/dcgm-dashboard.json
kubectl create configmap nvidia-dcgm-exporter-dashboard \
    -n rag \
    --from-file=nvidia-dcgm-exporter.json=/tmp/dcgm-dashboard.json \
    --dry-run=client -o yaml | \
    kubectl label --local -f - grafana_dashboard=1 --dry-run=client -o yaml | \
    kubectl apply -f -
```

이 대시보드는 Grafana의 사이드카에 의해 자동으로 로드되며 GPU 사용률, 온도, 메모리 사용량 및 기타 GPU 메트릭을 표시합니다.

---

**AI-Q Research Assistant 배포 (선택 사항)**

> **배포 선택**: 웹 검색 기능이 있는 자동화된 연구 보고서 생성이 필요한 경우 이러한 구성 요소를 배포하십시오. 사용 사례가 문서 Q&A용 Enterprise RAG Blueprint만 필요한 경우 [서비스 액세스](#서비스-액세스)로 진행하십시오.

<a id="6단계-ai-q-구성-요소-배포"></a>
### 6단계: AI-Q 구성 요소 배포

AI-Q Research Assistant를 배포합니다:

```bash
# TAVILY_API_KEY가 설정되었는지 확인
echo "Tavily API Key: ${TAVILY_API_KEY:0:10}..."

# NGC Helm 차트를 사용하여 AIRA 배포
helm upgrade --install aira https://helm.ngc.nvidia.com/nvidia/blueprint/charts/aiq-aira-v1.2.0.tgz \
  --username='$oauthtoken' \
  --password="${NGC_API_KEY}" \
  -n nv-aira --create-namespace \
  -f helm/aira-values.eks.yaml \
  --set imagePullSecret.password="$NGC_API_KEY" \
  --set ngcApiSecret.password="$NGC_API_KEY" \
  --set tavilyApiSecret.password="$TAVILY_API_KEY"
```

**대기 시간**: 70B 모델 다운로드에 15-20분

AI-Q 배포 확인:

```bash
# 모든 AIRA 구성 요소 확인
kubectl get all -n nv-aira

# 모든 구성 요소가 준비될 때까지 대기
kubectl wait --for=condition=ready pod -l app=aira -n nv-aira --timeout=1200s

# Pod 분포 확인
kubectl get pods -n nv-aira -o wide
```

</CollapsibleContent>

## 서비스 액세스

배포가 완료되면 포트 포워딩을 사용하여 로컬로 서비스에 액세스합니다.

<CollapsibleContent header={<h3><span>포트 포워딩 명령</span></h3>}>

**RAG 서비스용 포트 포워딩 시작:**

blueprints 디렉토리로 이동합니다:

```bash
cd ../../blueprints/inference/nvidia-deep-research
```

RAG 포트 포워딩 시작:

```bash
./app.sh port start rag
```

이를 통해 다음에 액세스할 수 있습니다:
- **RAG 프론트엔드**: http://localhost:3001 - RAG Q&A 직접 테스트
- **Ingestor API**: http://localhost:8082 - http://localhost:8082/docs에서 API 문서

**AI-Q 서비스용 포트 포워딩 시작** (배포된 경우):

```bash
./app.sh port start aira
```

이를 통해 다음에 액세스할 수 있습니다:
- **AIRA Research Assistant**: http://localhost:3000 - 웹 검색으로 포괄적인 연구 보고서 생성

**포트 포워딩 관리:**

상태 확인:
```bash
./app.sh port status
```

포트 포워딩 중지:
```bash
./app.sh port stop rag      # RAG 서비스 중지
./app.sh port stop aira     # AI-Q 서비스 중지
./app.sh port stop all      # 모든 서비스 중지
```

</CollapsibleContent>

### 애플리케이션 사용

**RAG 프론트엔드 (http://localhost:3001):**
- UI를 통해 직접 문서 업로드
- 수집된 문서에 대해 질문
- 다중 턴 대화 테스트
- 인용 및 소스 보기

**AI-Q Research Assistant (http://localhost:3000):**
- 연구 주제 및 질문 정의
- 업로드된 문서와 웹 검색 모두 활용
- 자동으로 포괄적인 연구 보고서 생성
- 다양한 형식으로 보고서 내보내기

**Ingestor API (http://localhost:8082/docs):**
- 프로그래매틱 문서 수집
- 배치 업로드 기능
- 컬렉션 관리
- OpenAPI 문서 보기

## 데이터 수집

RAG(및 선택적으로 AI-Q)를 배포한 후 OpenSearch 벡터 데이터베이스에 문서를 수집할 수 있습니다.

### 지원되는 파일 유형

RAG 파이프라인은 다음을 포함한 멀티모달 문서 수집을 지원합니다:
- PDF 문서
- 텍스트 파일 (.txt, .md)
- 이미지 (.jpg, .png)
- Office 문서 (.docx, .pptx)
- HTML 파일

NeMo Retriever 마이크로서비스는 이러한 문서에서 텍스트, 표, 차트 및 이미지를 자동으로 추출합니다.

### 수집 방법

문서를 수집하는 두 가지 옵션이 있습니다:

#### 방법 1: UI 업로드 (테스트/소규모 데이터셋)

프론트엔드 인터페이스를 통해 직접 개별 문서를 업로드합니다:

1. **RAG 프론트엔드** (http://localhost:3001) - 개별 문서 테스트에 이상적
2. **AIRA 프론트엔드** (http://localhost:3000) - 연구 작업용 문서 업로드

이 방법은 다음에 적합합니다:
- RAG 파이프라인 테스트
- 소규모 문서 컬렉션 (100개 미만)
- 빠른 실험
- 임시 문서 업로드

#### 방법 2: S3 배치 수집 (프로덕션/대규모 데이터셋)

<CollapsibleContent header={<h4><span>S3 배치 수집 명령</span></h4>}>

데이터 수집 스크립트를 사용하여 S3 버킷에서 문서를 배치 처리합니다. 권장 용도:
- 프로덕션 배포
- 대규모 문서 컬렉션 (수백에서 수천 개의 문서)
- 자동화된 수집 워크플로우
- 예약된 데이터 업데이트

**단계:**

1. RAG 포트 포워드가 실행 중인지 확인합니다:
   ```bash
   ./app.sh port start rag
   ```

2. 데이터 수집 스크립트를 실행합니다 (S3 버킷 세부 정보 입력 프롬프트):
   ```bash
   ./app.sh ingest
   ```

3. 또는 환경 변수를 설정하여 프롬프트 생략:
   ```bash
   export S3_BUCKET_NAME="your-pdf-bucket-name"
   export S3_PREFIX="documents/"  # 선택적 폴더 경로
   ./app.sh ingest
   ```

스크립트는 다음을 수행합니다:
- S3 버킷에서 문서 다운로드
- NVIDIA RAG 저장소에서 배치 수집 도구 다운로드
- NeMo Retriever 파이프라인을 통해 처리
- OpenSearch Serverless에 임베딩 저장
- 수집 진행 상황 및 통계 표시

> **추가 리소스**:
> - [RAG batch_ingestion.py 문서](https://github.com/NVIDIA-AI-Blueprints/rag/tree/v2.3.0/scripts)
> - [AI-Q 대량 데이터 수집 문서](https://github.com/NVIDIA-AI-Blueprints/aiq-research-assistant/blob/main/data/readme.md#bulk-upload-via-python)

</CollapsibleContent>

### 수집 확인

수집 후 문서가 사용 가능한지 확인합니다:

1. **RAG 프론트엔드를 통해**: [http://localhost:3001](http://localhost:3001) 로 이동하여 문서에 대해 질문
2. **Ingestor API를 통해**: http://localhost:8082/docs에서 컬렉션 통계 확인
3. **OpenSearch를 통해**: AWS 콘솔을 사용하여 OpenSearch 컬렉션에 직접 쿼리

## 관측성

RAG 및 AI-Q 배포에는 성능 모니터링, 요청 추적 및 메트릭 보기를 위한 내장 관측성 도구가 포함되어 있습니다.

### 모니터링 서비스 액세스

**자동화된 접근 방식 (권장):**

blueprints 디렉토리로 이동하고 포트 포워딩을 시작합니다:

```bash
cd ../../blueprints/inference/nvidia-deep-research
```

```bash
./app.sh port start observability
```

이것은 자동으로 포트 포워딩합니다:
- **Zipkin**: http://localhost:9411 - RAG 분산 추적
- **Grafana**: http://localhost:8080 - RAG 메트릭 및 대시보드
- **Phoenix**: http://localhost:6006 - AI-Q 워크플로우 추적 (배포된 경우)

상태 확인:
```bash
./app.sh port status
```

관측성 포트 포워딩 중지:
```bash
./app.sh port stop observability
```

<CollapsibleContent header={<h4><span>수동 kubectl 명령</span></h4>}>

**RAG 관측성 (Zipkin 및 Grafana):**

```bash
# 분산 추적을 위한 Zipkin 포트 포워딩 (별도 터미널에서 실행)
kubectl port-forward -n rag svc/rag-zipkin 9411:9411

# 메트릭 및 대시보드를 위한 Grafana 포트 포워딩 (다른 별도 터미널에서 실행)
kubectl port-forward -n rag svc/rag-grafana 8080:80
```

**AI-Q 관측성 (Phoenix):**

```bash
# AI-Q 추적을 위한 Phoenix 포트 포워딩 (별도 터미널에서 실행)
kubectl port-forward -n nv-aira svc/aira-phoenix 6006:6006
```

</CollapsibleContent>

### 모니터링 UI

포트 포워딩이 활성화되면:

- **Zipkin UI** (RAG 추적): http://localhost:9411
  - 엔드투엔드 요청 추적 보기
  - 지연 시간 병목 현상 분석
  - 다중 서비스 상호 작용 디버그

- **Grafana UI** (RAG 메트릭): http://localhost:8080
  - 기본 자격 증명: admin/admin
  - RAG 메트릭을 위한 사전 구축된 대시보드
  - GPU 사용률 및 처리량 모니터링

- **Phoenix UI** (AI-Q 추적): http://localhost:6006
  - 에이전트 워크플로우 시각화
  - LLM 호출 추적
  - 연구 보고서 생성 분석

> **참고**: 이러한 관측성 도구 사용에 대한 자세한 정보는 다음을 참조하십시오:
> - [Zipkin에서 추적 보기](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/observability.md#view-traces-in-zipkin)
> - [Grafana 대시보드에서 메트릭 보기](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/observability.md#view-metrics-in-grafana)

> **대안**: 모니터링 서비스를 공개적으로 노출해야 하는 경우 적절한 인증 및 보안 제어가 있는 Ingress 리소스를 생성할 수 있습니다.

## 정리

### 애플리케이션만 제거

인프라를 유지하면서 RAG 및 AI-Q 애플리케이션을 제거하려면:

**자동화 스크립트 사용 (권장):**

```bash
cd ../../blueprints/inference/nvidia-deep-research
```

```bash
./app.sh cleanup
```

정리 스크립트는 다음을 수행합니다:
- 모든 포트 포워딩 프로세스 중지
- AIRA 및 RAG Helm 릴리스 제거
- 로컬 포트 포워딩 PID 파일 제거

**수동 애플리케이션 정리:**

```bash
# blueprints 디렉토리로 이동
cd ../../blueprints/inference/nvidia-deep-research

# 포트 포워딩 중지
./app.sh port stop all

# AIRA 제거 (배포된 경우)
helm uninstall aira -n nv-aira

# RAG 제거
helm uninstall rag -n rag
```

**(선택 사항) 배포 중에 생성된 임시 파일 정리:**

```bash
rm /tmp/.port-forward-*.pid
```

> **참고**: 이것은 애플리케이션만 제거합니다. EKS 클러스터와 인프라는 계속 실행됩니다. GPU 노드는 5-10분 내에 [Karpenter](https://karpenter.sh/)에 의해 종료됩니다.

### 인프라 정리

전체 EKS 클러스터 및 모든 인프라 구성 요소를 제거하려면:

```bash
# infra 디렉토리로 이동
cd ../../../infra/nvidia-deep-research

# 정리 스크립트 실행
./cleanup.sh
```

> **경고**: 이것은 영구적으로 삭제합니다:
> - EKS 클러스터 및 모든 워크로드
> - OpenSearch Serverless 컬렉션 및 데이터
> - VPC 및 네트워킹 리소스
> - 모든 관련 AWS 리소스
>
> 진행하기 전에 중요한 데이터를 백업하십시오.

**소요 시간**: 전체 해제에 ~10-15분

## 비용 고려 사항

<CollapsibleContent header={<h3><span>이 배포의 예상 비용</span></h3>}>

:::warning 중요
GPU 인스턴스와 지원 인프라는 실행 상태로 유지되면 상당한 비용이 발생할 수 있습니다. 예상치 못한 요금을 피하려면 **사용하지 않을 때는 항상 리소스를 정리하십시오.**
:::

<a id="예상-월간-비용"></a>
### 예상 월간 비용

다음 표는 US West 2 (Oregon) 리전의 **기본 배포**에 대한 대략적인 비용을 보여줍니다. 실제 비용은 리전, 사용 패턴 및 워크로드 기간에 따라 달라집니다.

| 리소스 | 구성 | 예상 월간 비용 | 참고 |
|--------|--------------|----------------------|-------|
| **EKS Control Plane** | 1 클러스터 | **~$73/월** | 고정 비용: $0.10/시간 x 730시간 |
| **GPU 인스턴스 (RAG만)** | 1x g5.48xlarge (8x A10G)<br/>2x g5.12xlarge (각 4x A10G) | **~$20,171/월*** | 워크로드 실행 중에만<br/>유휴 시 Karpenter가 축소 |
| **GPU 인스턴스 (RAG + AI-Q)** | 추가 g5.48xlarge | **~$32,061/월*** | 추가 70B 모델에 8개 더 많은 GPU 필요 |
| **OpenSearch Serverless** | 2-4 OCU (일반적) | **~$350-700/월** | $0.24/OCU-시간<br/>데이터 볼륨 및 쿼리에 따라 확장 |
| **NAT Gateway** | 2 AZ | **~$66/월** | 고정: 2 게이트웨이 x $0.045/시간 x 730시간<br/>추가 데이터 처리: $0.045/GB |
| **ECR Storage** | Docker 이미지 | **~$5-10/월** | 50-100GB 사용자 정의 이미지<br/>ECR 가격: $0.10/GB/월 |
| **EBS Volumes** | 노드 스토리지 | **~$72/월** | 노드당 300GB gp3 x 3 노드 x $0.08/GB<br/>GPU 노드 실행 중에만 청구 |
| **Data Transfer** | Cross-AZ, 인터넷 | **가변** | 사용 패턴에 따라 다름<br/>Cross-AZ: $0.01/GB, 인터넷: $0.09/GB |

**\*GPU 인스턴스 비용은 연속 운영을 가정합니다. 아래 세부 정보를 참조하십시오.**

<a id="gpu-인스턴스-비용-세부-정보"></a>
### GPU 인스턴스 비용 세부 정보

GPU 인스턴스는 **주요 비용 요소**입니다. 비용은 인스턴스 유형과 실행 기간에 따라 달라집니다:

**기본 구성 (G5 인스턴스 - RAG만):**

| 인스턴스 유형 | GPU | 온디맨드 요금 | 일일 비용 (24시간) | 월간 비용 (730시간) |
|---------------|------|----------------|-------------------|---------------------|
| g5.48xlarge (x1) | 8x A10G | $16.288/시간 | $390.91 | $11,890.24 |
| g5.12xlarge (x2) | 각 4x A10G | 각 $5.672/시간 | 각 $136.13 | 각 $4,140.56 |

**RAG 총계**: 24/7 실행 시 ~$20,171/월 (1x g5.48xlarge + 2x g5.12xlarge = $11,890 + $8,281)

**AI-Q 포함 (추가 70B 모델):**
- 추가 g5.48xlarge: $11,890.24/월
- **총계**: 24/7 실행 시 ~$32,061/월 (2x g5.48xlarge + 2x g5.12xlarge)

> **참고**: 대체 인스턴스 유형(G6e, P4, P5)을 사용하는 경우 비용이 달라집니다. 리전 및 인스턴스 유형에 대해 [AWS EC2 가격](https://aws.amazon.com/ec2/pricing/on-demand/)을 확인하십시오.

</CollapsibleContent>

## 참조

### 공식 NVIDIA 리소스

**문서:**
- [NVIDIA AI-Q Research Assistant GitHub](https://github.com/NVIDIA-AI-Blueprints/aiq-research-assistant): 공식 AI-Q 블루프린트 저장소
- [NVIDIA AI-Q on AI Foundation](https://build.nvidia.com/nvidia/aiq): AI-Q 블루프린트 카드 및 호스팅 버전
- [NVIDIA RAG Blueprint](https://github.com/NVIDIA-AI-Blueprints/rag): 완전한 RAG 플랫폼 문서
- [NVIDIA NIM Documentation](https://docs.nvidia.com/nim/): NIM 마이크로서비스 참조
- [NVIDIA AI Enterprise](https://www.nvidia.com/en-us/data-center/products/ai-enterprise/): 엔터프라이즈 AI 플랫폼

**모델:**
- [Llama-3.3-Nemotron-Super-49B-v1.5](https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1_5): 고급 추론 모델 (490억 파라미터)
- [Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct): 명령어 따르기 모델

**컨테이너 이미지 및 Helm 차트:**
- [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/): 공식 컨테이너 레지스트리
- [RAG Blueprint Helm Chart](https://helm.ngc.nvidia.com/nvidia/blueprint/charts/nvidia-blueprint-rag): Kubernetes 배포
- [NVIDIA NIM Containers](https://catalog.ngc.nvidia.com/orgs/nim): 최적화된 추론 컨테이너

### AI-on-EKS 블루프린트 리소스

**AI-on-EKS 블루프린트 리소스:**
- [AI-on-EKS Repository](https://github.com/awslabs/ai-on-eks): 메인 블루프린트 저장소
- [Infrastructure & Deployment Code](https://github.com/awslabs/ai-on-eks/tree/main/infra/nvidia-deep-research): Karpenter와 애플리케이션 배포 스크립트를 포함한 Terraform 자동화
- [Usage Guide](https://github.com/awslabs/ai-on-eks/tree/main/blueprints/inference/nvidia-deep-research): 배포 후 사용, 데이터 수집 및 관측성

**문서:**
- [Infrastructure & Deployment Guide](https://github.com/awslabs/ai-on-eks/tree/main/infra/nvidia-deep-research/README.md): 단계별 인프라 및 애플리케이션 배포
- [Usage Guide](https://github.com/awslabs/ai-on-eks/tree/main/blueprints/inference/nvidia-deep-research/README.md): 서비스 액세스, 데이터 수집, 모니터링
- [OpenSearch Integration](https://github.com/awslabs/ai-on-eks/tree/main/infra/nvidia-deep-research/terraform/opensearch-serverless.tf): Pod Identity 인증 설정
- [Karpenter Configuration](https://github.com/awslabs/ai-on-eks/tree/main/infra/nvidia-deep-research/terraform/custom_karpenter.tf): P4/P5 GPU 지원

### 관련 기술

**Kubernetes 및 AWS:**
- [Amazon EKS](https://aws.amazon.com/eks/): 관리형 Kubernetes 서비스
- [Karpenter](https://karpenter.sh/): Kubernetes 노드 오토스케일링
- [OpenSearch Serverless](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/serverless.html): 관리형 벡터 데이터베이스
- [EKS Pod Identity](https://docs.aws.amazon.com/eks/latest/userguide/pod-identities.html): Pod용 IAM 인증

**AI/ML 도구:**
- [NVIDIA DCGM](https://developer.nvidia.com/dcgm): GPU 모니터링
- [Prometheus](https://prometheus.io/): 메트릭 수집
- [Grafana](https://grafana.com/): 시각화 대시보드

## 다음 단계

1. **기능 탐색**: 다양한 파일 유형으로 멀티모달 문서 처리 테스트
2. **배포 확장**: 다중 리전 또는 다중 클러스터 설정 구성
3. **애플리케이션 통합**: 애플리케이션을 RAG API 엔드포인트에 연결
4. **성능 모니터링**: 지속적인 모니터링을 위해 Grafana 대시보드 사용
5. **사용자 정의 모델**: 자체 미세 조정된 모델로 교체
6. **보안 강화**: 인증, 속도 제한 및 재해 복구 추가

---

이 배포는 [NVIDIA Enterprise RAG Blueprint](https://github.com/NVIDIA-AI-Blueprints/rag) 및 [NVIDIA AI-Q Research Assistant](https://github.com/NVIDIA-AI-Blueprints/aiq-research-assistant)를 [Karpenter](https://karpenter.sh/) 자동 확장, OpenSearch Serverless 통합 및 원활한 AWS 서비스 통합을 포함한 엔터프라이즈급 기능과 함께 Amazon EKS에 제공합니다.
