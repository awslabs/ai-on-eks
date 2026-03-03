# 관측성

AI/ML 워크로드의 관측성은 로그, 메트릭, 트레이스와 같은 여러 데이터 소스와 함께 여러 하드웨어/소프트웨어 구성 요소에 대한 전체적인 시각이 필요합니다. 이러한 구성 요소들을 조합하는 것은 어렵고 시간이 많이 소요되므로, 이 환경을 부트스트랩하기 위해 Github에서 제공하는 [AI/ML 관측성](https://github.com/awslabs/ai-ml-observability-reference-architecture)을 활용합니다.

## 아키텍처
![아키텍처](https://github.com/awslabs/ai-ml-observability-reference-architecture/raw/main/static/reference_architecture.png)

## 포함 내용
- Prometheus
- OpenSearch
- FluentBit
- Kube State Metrics
- Metrics Server
- Alertmanager
- Grafana
- AI/ML 워크로드를 위한 Pod/Service 모니터
- AI/ML 대시보드

## 필요성

AI/ML 워크로드의 성능을 이해하는 것은 어렵습니다: GPU가 충분히 빠르게 데이터를 받고 있는가? CPU가 병목인가? 스토리지가 충분히 빠른가? 이러한 질문들은 개별적으로 답하기 어렵습니다. 전체 그림을 더 많이 볼 수 있을수록 성능 병목을 식별하는 데 더 명확해집니다.

## 사용 방법

[JARK](https://awslabs.github.io/ai-on-eks/docs/infra/jark) 인프라에는 이 아키텍처가 기본적으로 활성화되어 있습니다. 인프라에 추가하려면 `blueprint.tfvars`에서 2개의 변수를 `true`로 설정해야 합니다:

```yaml
enable_argocd                    = true
enable_ai_ml_observability_stack = true
```

첫 번째 변수는 관측성 아키텍처를 배포하는 데 사용되는 ArgoCD를 배포하고, 두 번째 변수는 아키텍처를 배포합니다.

## 사용법

아키텍처는 전적으로 `monitoring` 네임스페이스에 배포됩니다. Grafana에 접근하려면: `kubectl port-forward -n monitoring svc/kube-prometheus-stack-grafana 3000:80`. 그런 다음 [https://localhost:3000](https://localhost:3000) 을 열어 사용자 이름 `admin`과 비밀번호 `prom-operator`로 grafana에 로그인할 수 있습니다. 사용자 이름/비밀번호를 변경하는 방법은 Readme의 [보안](https://github.com/awslabs/ai-ml-observability-reference-architecture?tab=readme-ov-file#security) 섹션을 참조하십시오.

### 훈련

Ray 훈련 작업 로그와 메트릭은 관측성 아키텍처에 의해 자동으로 수집되며 [훈련 대시보드](http://localhost:3000/d/ee6mbjghme96oc/gpu-training?orgId=1&refresh=5s&var-namespace=default&var-job=ray-train&var-instance=All)에서 확인할 수 있습니다.

#### 예제

이에 대한 전체 예제는 [AI/ML 관측성 저장소](https://github.com/awslabs/ai-ml-observability-reference-architecture/tree/main/examples/training)에서 찾을 수 있습니다. 또한 여기의 블루프린트도 이 아키텍처를 활용하도록 업데이트할 예정입니다.

### 추론

Ray 추론 메트릭은 관측성 인프라에 의해 자동으로 수집되어야 하며 [추론 대시보드](http://localhost:3000/d/bec31e71-3ac5-4133-b2e3-b9f75c8ab56c/inference-dashboard?orgId=1&refresh=5s)에서 확인할 수 있습니다. 추론 워크로드를 로깅을 위해 계측하려면 몇 가지 항목을 추가해야 합니다:

#### FluentBit 구성

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentbit-config
  namespace: default
data:
  fluent-bit.conf: |-
    [INPUT]
        Name tail
        Path /tmp/ray/session_latest/logs/*
        Tag ray
        Path_Key true
        Refresh_Interval 5
    [FILTER]
        Name modify
        Match ray
        Add POD_LABELS ${POD_LABELS}
    [OUTPUT]
        Name stdout
        Format json
```

추론 워크로드를 실행할 네임스페이스에 이것을 배포합니다. FluentBit 사이드카에 로그를 출력하는 방법을 알려주기 위해 각 네임스페이스에 하나만 필요합니다.

#### FluentBit 사이드카

FluentBit가 로그를 STDOUT에 쓸 수 있도록 Ray 추론 서비스에 사이드카를 추가해야 합니다.

```yaml
              - name: fluentbit
                image: fluent/fluent-bit:3.2.2
                env:
                  - name: POD_LABELS
                    valueFrom:
                      fieldRef:
                        fieldPath: metadata.labels['ray.io/cluster']
                resources:
                  requests:
                    cpu: 100m
                    memory: 128Mi
                  limits:
                    cpu: 100m
                    memory: 128Mi
                volumeMounts:
                  - mountPath: /tmp/ray
                    name: ray-logs
                  - mountPath: /fluent-bit/etc/fluent-bit.conf
                    subPath: fluent-bit.conf
                    name: fluentbit-config
```

이 섹션을 `workerGroupSpecs` 컨테이너에 추가합니다.

#### FluentBit 볼륨

마지막으로 configmap 볼륨을 `volumes` 섹션에 추가해야 합니다:

```yaml
              - name: fluentbit-config
                configMap:
                  name: fluentbit-config
```

#### vLLM 메트릭

vLLM은 또한 Time to First Token, 처리량, 지연 시간, 캐시 활용도 등과 같은 유용한 메트릭을 출력합니다. 이러한 메트릭에 접근하려면 메트릭 경로를 위한 라우트를 Pod에 추가해야 합니다:

```python
# Imports
import re
from prometheus_client import make_asgi_app
from fastapi import FastAPI
from starlette.routing import Mount

app = FastAPI()

class Deployment:
    def __init__(self, **kwargs):
        ...
        route = Mount("/metrics", make_asgi_app())
        # Workaround for 307 Redirect for /metrics
        route.path_regex = re.compile('^/metrics(?P<path>.*)$')
        app.routes.append(route)
```

이렇게 하면 배포된 모니터가 vLLM 메트릭을 수집하고 추론 대시보드에 표시할 수 있습니다.

#### 예제

이에 대한 전체 예제는 [AI/ML 관측성 저장소](https://github.com/awslabs/ai-ml-observability-reference-architecture/tree/main/examples/inference)에서 찾을 수 있습니다. 또한 여기의 블루프린트도 이 아키텍처를 활용하도록 업데이트할 예정입니다.
