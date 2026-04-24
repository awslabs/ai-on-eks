---
sidebar_label: EKS에서의 Agentic AI
---

# EKS에서의 Agentic AI

이 섹션은 Amazon EKS에서 AI 에이전트를 구축, 배포 및 운영하기 위한 가이드를 제공합니다. 소스 제어, CI/CD, 관측성, 벡터 스토리지, MCP 도구 관리를 통합 인프라로 결합한 오픈 소스 환경인 [Agents on EKS](../../infra/agents-on-eks.md) 레퍼런스 환경을 중심으로 구성되어 있습니다.

## 대상

로컬 개발 환경을 넘어 에이전트를 배포하려는 팀을 위한 가이드입니다. 첫 번째 에이전트를 배포하거나 에이전트 변경 사항을 지속적으로 테스트하고 프로모션하는 파이프라인을 구축하는 경우, 이 가이드는 Kubernetes의 오픈 소스 도구를 사용한 실용적인 단계를 안내합니다.

## 학습 내용

- [에이전트 개발 모범 사례](./best-practices.md) — 에이전트 코드를 로컬 환경에서 온라인 환경으로 원활하게 전환할 수 있도록 구조화하는 패턴입니다. 의존성 관리, 테스트 가능성을 위한 호출 로직 분리, 에이전트를 REST API로 래핑하기, 확률적 출력을 처리하는 AgentOps 철학을 다룹니다.

- [에이전트 구축 및 배포](./building-agents.md) — 환경 사용에 대한 집중 안내: 에이전트 컨테이너화, GitLab에 코드 푸시, 자동 이미지 빌드를 위한 CI/CD 설정, Kubernetes 배포, Pod Identity를 통한 AWS 액세스 구성.

## 환경

Agents on EKS 인프라는 다음 구성 요소를 EKS 클러스터에 배포합니다:

| 구성 요소 | 용도 |
|-----------|------|
| [GitLab](https://about.gitlab.com/) | 소스 제어, 컨테이너 레지스트리, CI/CD 파이프라인 |
| [LangFuse](https://langfuse.com/) | LLM 관측성, 트레이싱, 평가 |
| [Milvus](https://milvus.io/) | 임베딩 및 에이전트 메모리를 위한 벡터 데이터베이스 |
| [MCP Gateway Registry](https://github.com/agentic-community/mcp-gateway-registry) | MCP 서버 검색 및 관리 |

배포 지침 및 구성 옵션은 [인프라 가이드](../../infra/agents-on-eks.md)를 참조하세요.
