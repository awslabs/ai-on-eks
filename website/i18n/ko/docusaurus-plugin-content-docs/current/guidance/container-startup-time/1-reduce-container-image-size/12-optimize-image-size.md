---
sidebar_label: 컨테이너 이미지 크기 최적화
---

# 컨테이너 이미지 크기 최적화

## 적절한 베이스 이미지 선택

다양한 AI/ML 프레임워크와 플랫폼은 편의성을 제공하고 실험을 가능하게 하는 즉시 사용 가능한 컨테이너 이미지를 제공합니다. 그러나 이러한 이미지는 가능한 한 넓은 기능 세트를 다루려고 하므로 다양한 런타임, 프레임워크 또는 지원되는 API를 포함할 수 있어 비대화로 이어질 수 있습니다.

예를 들어, 다양한 PyTorch 이미지 변형은 매우 다른 크기를 가집니다: 개발 도구, 컴파일러 등을 포함하는 [2.7.1-cuda11.8-cudnn9-devel](https://hub.docker.com/layers/pytorch/pytorch/2.7.1-cuda11.8-cudnn9-devel/images/sha256-5a046e4e3364b063a17854387b8820ad3f42ed197a089196bce8f2bd68f275a8) (6.66 GB)부터 런타임만 포함하는 [2.7.1-cuda11.8-cudnn9-runtime](https://hub.docker.com/layers/pytorch/pytorch/2.7.1-cuda11.8-cudnn9-runtime/images/sha256-8d409f72f99e5968b5c4c9396a21f4b723982cfdf2c1a5b9cc045c5d0a7345a1) (3.03 GB)까지 있습니다. vLLM 프로젝트는 OpenAI 사양, Sagemaker 통합 등과 같은 다양한 기능이 패키징된 여러 컨테이너 이미지 변형을 [제공](https://docs.vllm.ai/en/stable/contributing/dockerfile/dockerfile.html)합니다.

애플리케이션의 필요를 충족하는 더 작은 베이스 이미지를 선택하면 큰 차이를 만들 수 있습니다. 주의할 점은 더 작은 런타임 전용 이미지에는 JIT 컴파일이나 동적 최적화가 포함되지 않아 더 느린 코드 경로에 빠져 시작 시간이 줄어들 수 있다는 것입니다.

포괄적인 접근 방식은 다음을 포함합니다:

* 다양한 베이스 이미지로 워크로드 벤치마킹
* 필요한 최적화 라이브러리만 포함하는 사용자 정의 빌드 고려
* 이미지 풀 시간 개선 외에 전체 콜드 스타트 성능 테스트

## 멀티 스테이지 빌드 사용

Docker/BuildKit, Podman, Finch/Buildkit과 같은 여러 플랫폼에서 지원하는 멀티 스테이지 컨테이너 이미지 빌드를 통해 단일 컨테이너 이미지 파일에서 여러 FROM 문을 사용하여 빌드 프로세스와 아티팩트를 런타임 관심사와 분리할 수 있습니다.

멀티 스테이지 빌드 컨테이너 이미지 파일은 다음과 유사할 수 있습니다:

```
# 외부 이미지를 사용하여 아티팩트를 복사하는 빌드 스테이지
FROM python:3.12-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.7.11 /uv /uvx /bin/
...

# 런타임 스테이지
FROM python:3.12-slim-bookworm
...
COPY --from=models some-model /app/models/some-model/configs
COPY --from=builder --chown=app:app /app/.venv ./.venv
COPY --from=builder --chown=app:app /app/main.py ./main.py
...
CMD ["sh", "-c", "exec fastapi run --host 0.0.0.0 --port 80 /app/main.py"]
```

:::info
일반적인 애플리케이션 종속성을 컨테이너 이미지에 베이킹하는 것과 달리, 대규모 모델 파일(수 GB에서 수십 GB 범위)을 복사하는 것은 일반적으로 권장되지 않습니다. 이는 풀 시간에 영향을 미치는 컨테이너 이미지 크기 증가, 앱과 모델에 대한 별도의 릴리스 라이프사이클, 여러 앱 간에 모델을 공유할 때 잠재적인 스토리지 중복 때문입니다.
:::

필요한 아티팩트만 복사하면 빌드 결과의 어떤 구성 요소가 최종 런타임 이미지에 포함될지 세밀하게 제어할 수 있어 크기가 줄어듭니다(보안이나 워크플로우 단순성과 같은 다른 이점과 함께).

위의 예에서 우리는 `COPY --from`의 두 가지 다른 [변형](https://docs.docker.com/reference/dockerfile/#copy---from)(대부분의 인기 있는 이미지 빌드 플랫폼에서 BuildKit을 통해 지원됨)도 사용했습니다:

* `COPY --from=<레지스트리의 이미지 경로와 가져올 부분>` 레지스트리에 저장된 다른 컨테이너 이미지에서 특정 파일과 폴더만 추출할 수 있습니다
* `COPY --from=<빌드 컨텍스트 이름>` `--build-context models=/path/to/local/folder`를 사용하여 빌드 명령에 매개변수로 제공된 로컬 폴더에서 특정 파일과 폴더만 복사할 수 있습니다

`.dockerignore`를 사용하는 것은 일반적으로 좋은 관행이며 위의 프로세스와 함께 사용해야 하지만 `COPY --from=...` 명령에는 영향을 미치지 않습니다.

또한 이 기법은 다음을 사용하여 더욱 개선할 수 있지만(때로는 미미하게), 주의 사항을 고려해야 합니다.

## 레이어 최적화 기법 적용

이미지 레이어(이제 총 크기가 더 작음)가 풀 프로세스 중에 다운로드되면 컨테이너의 파일 시스템을 조립하기 위해 압축이 해제되고 언팩됩니다. 이미지 레이어의 양과 크기는 해당 프로세스의 기간에 영향을 미쳐 최적화의 또 다른 후보가 됩니다.

일반적으로 언급되는 최적화 중 하나는 `RUN` 또는 `COPY` 명령을 결합하여 더 적은 수의 더 큰 레이어를 만드는 것으로, 다음과 같은 일반적인 예가 있습니다:

```
FROM ...
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libssl-dev \
        pkg-config && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean
```

멀티 스테이지 빌드를 적용하라는 권장 사항의 맥락에서 `RUN`은 마지막 런타임 스테이지의 일부가 아닌 경우가 많습니다. 이상적으로는 모든 실행이 이전 빌드 스테이지에서 수행되고 처리된 아티팩트만 런타임 스테이지의 의도된 위치에 복사되기 때문입니다.

런타임 스테이지의 `COPY` 명령은 이 프로세스를 사용하여 최적화할 수 있지만, 한 가지 문제가 있습니다. 여러 대상이나 여러 소스 스테이지를 지원하지 않으므로 이러한 명령을 하나로 결합하는 것이 불가능합니다:

```
COPY --from=models some-model /app/models/some-model/configs
COPY --from=builder1 --chown=app:app /app/.venv ./.venv
COPY --from=builder1 --chown=app:app /app/main.py ./main.py
COPY --from=builder2 --chown=app:app /app/config.json ./config.json
```

이를 극복하기 위해 최종 폴더 구조가 생성된 후 단일 명령을 통해 복사되는 추가 복사 스테이지를 도입할 수 있습니다:

```
FROM python:3.12-bookworm AS builder1
WORKDIR /app
...

FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel AS builder2
WORKDIR /app
...

FROM scratch AS assembly
...
COPY --from=models some-model/weights /app/models/some-model/weights
COPY --from=builder1 /app/.venv /app/venv
COPY --from=builder1 /app/main.py /app/main.py
COPY --from=builder2 /dist/config.json /app/config.json

FROM python:3.12-slim-bookworm
COPY --from=assembly --chown=app:app /app /app
CMD ["python", "main.py"]
```

위의 단계가 전체 컨테이너 시작 시간에 긍정적인 영향을 미치더라도 이 가이드의 다른 솔루션에 비해 종종 무시할 수 있으며 기법에 시간을 투자하기 전에 평가해야 합니다.

개선 사항은 다음을 포함하는 트레이드오프에 대해 가중치를 부여해야 합니다:

* 레이어가 올바르게 정렬되지 않은 경우 더 적은 수의 더 큰 레이어로 인한 더 낮은 세분성으로 캐시 효율성 감소
* 런타임 레이어 최적화를 위한 더 많은 셔플링으로 인한 빌드 시간
