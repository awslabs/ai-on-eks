---
sidebar_label: 에이전트 개발 모범 사례
---

# 에이전트 개발 모범 사례

이 가이드는 로컬 개발에서 [Agents on EKS 환경](../../infra/agents-on-eks.md)의 프로덕션으로 원활하게 전환할 수 있는 에이전트 개발 패턴과 모범 사례를 다룹니다. [Strands Agents SDK](https://strandsagents.com)를 사용한 실용적인 예제를 통해 각 패턴을 설명합니다.

## AgentOps가 필요한 이유

대규모로 AI 에이전트를 구축하고 운영하려면 추론 인프라 이상의 것이 필요합니다. 에이전트 출력의 확률적 특성에 맞게 기존 DevOps 관행을 적응시켜야 합니다. 주요 과제는 다음과 같습니다:

- **확률적 테스트**: `expected == actual` 테스트는 LLM 출력에 적합하지 않습니다. 평가에는 유연성이 필요합니다.
- **데이터셋 큐레이션**: 단일 턴 vs 다중 턴 평가는 서로 다른 접근 방식이 필요합니다.
- **복합 메트릭**: 정확도, 도구 선택, 응답 품질, 지연 시간, 토큰 활용도 모두 에이전트 성능 평가에 영향을 미칩니다.
- **지속적 평가**: 배포된 에이전트는 배포 전 테스트뿐만 아니라 지속적인 모니터링이 필요합니다.
- **샌드박스 배포**: 새 에이전트 버전은 라이브 에이전트를 대체하기 전에 격리된 환경에서 테스트해야 합니다.

AgentOps 파이프라인은 다음 흐름으로 이러한 과제를 해결합니다:

**소스 제어 → 이미지 빌드 → 테스트 배포 → 테스트 데이터셋 → 메트릭 확인 → 에이전트 재배포 → 테스트 제거**

이를 통해 모든 코드 변경이 빌드되고, 샌드박스에 배포되며, 데이터셋을 기준으로 평가되고, 품질 기준을 충족하는 경우에만 프로덕션에 프로모션됩니다.

## 개발 환경 설정

:::note
에이전트마다 별도의 Python 가상 환경을 사용하는 것을 권장합니다. 최종적으로 격리된 환경에서 빌드되므로, 가상 환경을 사용하면 로컬 머신에서 구축한 에이전트가 프로덕션 환경에서도 유사하게 작동하도록 보장합니다. `venv`는 Python 3.3+ 버전에 포함되어 있으며 이 문서에서 사용됩니다. conda나 다른 도구를 사용해도 무방합니다.
:::

```bash
mkdir -p ~/code/strands-agent && cd ~/code/strands-agent
python -m venv .venv
source .venv/bin/activate
```

## 패턴: requirements.txt로 시작하기

`requirements.txt` 파일은 Python 의존성 설치를 지원합니다. pip을 사용하여 임시로 라이브러리를 설치하기보다 이 파일을 먼저 사용하는 것을 권장합니다. 임시 설치는 다른 Python 환경으로 이동하거나 프로덕션용으로 빌드할 때 동기화 문제가 발생하기 쉽습니다.

```text
strands-agents>=1.0.0
```

```bash
pip install -r requirements.txt
```

이 파일은 에이전트 의존성의 단일 진실 원천(single source of truth)이 되며, 프로덕션용 빌드 시 Dockerfile에서 직접 사용됩니다.

## 예제: 간단한 번역 에이전트

패턴을 설명하기 위해 [Amazon Bedrock](https://aws.amazon.com/bedrock/)을 사용한 [Strands Agents SDK](https://strandsagents.com)로 간단한 번역 에이전트를 구축해 보겠습니다:

```python
from strands import Agent


class MyAgent:
    def __init__(self):
        self.agent = Agent(
            model="us.amazon.nova-lite-v1:0",
            system_prompt="You are an expert translator. Translate the user request from English into Italian. Do not respond with anything other than the translation"
        )

    def invoke(self, request: str) -> str:
        result = self.agent(request)
        return result.message["content"][0]["text"]

```

획기적인 에이전트는 아니지만, 아키텍처를 재설계하지 않고도 로컬과 프로덕션 모두에서 작동하는 패턴을 보여주기에 충분합니다.

## 패턴: 호출 로직과 에이전트 분리

호출 로직을 에이전트에서 분리하면 더 복잡한 로직을 만들 수 있을 뿐만 아니라 테스트 및 사용을 위한 간편한 진입점을 제공합니다.

`agent.py`:

```python
from strands import Agent


class MyAgent:
    def __init__(self):
        self.agent = Agent(
            model="us.amazon.nova-lite-v1:0",
            system_prompt="You are an expert translator. Translate the user request from English into Italian. Do not respond with anything other than the translation"
        )

    def invoke(self, request: str) -> str:
        result = self.agent(request)
        return result.message["content"][0]["text"]

```

`agent_entry.py`:

```python
from agent import MyAgent

agent = MyAgent()


def request(request: str):
    return agent.invoke(request)


print(request("Hello!"))
```

이를 통해 라이브러리와 유사하게 에이전트를 다양한 용도로 활용할 수 있습니다. 테스트를 별도로 작성할 수 있습니다:

`test_agent.py`:

```python
import unittest
from agent import MyAgent


class TestMyAgent(unittest.TestCase):
    def test_translate(self):
        agent = MyAgent()
        result = agent.invoke("Hello!")
        self.assertEqual(result, "Ciao!")


if __name__ == '__main__':
    unittest.main()
```

`python test_agent.py`로 실행합니다.

이 테스트는 간단하지만 코드 변경으로 인한 드리프트를 감지할 수 있습니다 (예: 프롬프트를 이탈리아어 대신 스페인어로 변경하는 경우):

```text
AssertionError: '¡Hola!' != 'Ciao!'
- ¡Hola!
+ Ciao!
```

## 패턴: 에이전트를 API로 만들기

현재 에이전트는 고정된 입력으로 Python 스크립트를 실행합니다. 프로덕션에서 유용하게 사용하려면:

- 하드코딩된 입력 대신 사용자 입력을 받아야 합니다
- REST API로 동작해야 합니다

다음은 에이전트를 FastAPI 엔드포인트로 래핑하는 방법입니다:

`agent.py`:

```python
from strands import Agent


class MyAgent:
    def __init__(self):
        self.agent = Agent(
            model="us.amazon.nova-lite-v1:0",
            system_prompt="You are an expert translator. Translate the user request from English into Italian. Do not respond with anything other than the translation"
        )

    def invoke(self, request: str) -> str:
        result = self.agent(request)
        return result.message["content"][0]["text"]

```

`agent_entry.py` (범용 API 래퍼)

```python
from agent import MyAgent
from api_manager import endpoint
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import json
import asyncio

agent = MyAgent()


class Request(BaseModel):
    request: str


@endpoint(method="post")
async def request(request: Request):
    return agent.invoke(request.request)

```

`entrypoint.py`:

```python
import uvicorn
from agent import app

uvicorn.run(app, host="0.0.0.0", port=8000)

```

`requirements.txt` 업데이트:

```text
strands-agents>=1.0.0
fastapi==0.116.1
uvicorn==0.35.0
```

:::note
재사용성을 위해 `api_manager.py`, `agent_entry.py`, `entrypoint.py`를 `agent.py`와 분리합니다. 처음 3개의 파일은 범용이며 향후 에이전트를 스캐폴딩할 수 있게 해주고, `agent.py`는 로직을 캡슐화하고 확장할 수 있는 스크립트로 유지됩니다. 코드 구조는 자유롭게 구성하셔도 됩니다. 컨테이너화 섹션에서 이 패턴을 계속 발전시킬 것입니다.
:::

테스트:

```bash
python entrypoint.py
```

```bash
curl --location 'http://localhost:8000/' \
--header 'Content-Type: application/json' \
--data '{"request": "Hello"}'
```

```text
"Ciao"
```

## 요약

이 시점에서 에이전트는 다음과 같은 상태입니다:

1. 일관된 클린 환경을 위한 `requirements.txt`와 가상 환경 사용
2. 테스트 가능성을 위해 호출 로직이 메인 스크립트에서 분리됨
3. 드리프트를 감지하는 간단한 테스트 보유
4. 프로덕션 사용을 위한 REST API로 래핑됨

이 에이전트는 이제 Agents on EKS 환경에 [컨테이너화하고 배포](./building-agents.md)할 준비가 되었습니다.

## 다음 단계

- [에이전트 구축 및 배포](./building-agents.md) — 컨테이너화, GitLab 푸시, Kubernetes 배포
