---
sidebar_label: EKS의 AIBrix
---
import CollapsibleContent from '@site/src/components/CollapsibleContent';


# AIBrix

AIBrix는 확장 가능한 생성형 AI 추론(Inference) 인프라를 구축하기 위한 필수 빌딩 블록을 제공하도록 설계된 오픈 소스 이니셔티브입니다. AIBrix는 엔터프라이즈 요구에 맞춰 대규모 언어 모델(LLM) 추론 배포, 관리 및 확장에 최적화된 클라우드 네이티브 솔루션을 제공합니다.
![Alt text](https://aibrix.readthedocs.io/latest/_images/aibrix-architecture-v1.jpeg)

### 기능
* LLM 게이트웨이 및 라우팅: 여러 모델과 레플리카에 걸쳐 트래픽을 효율적으로 관리하고 전달합니다.
* 고밀도 LoRA 관리: 모델의 경량 저순위 적응(Low-Rank Adaptation)을 간소화된 방식으로 지원합니다.
* 분산 추론: 여러 노드에 걸쳐 대규모 워크로드를 처리할 수 있는 확장 가능한 아키텍처입니다.
* LLM 앱 맞춤형 오토스케일러: 실시간 수요에 따라 추론 리소스를 동적으로 확장합니다.
* 통합 AI 런타임: 메트릭 표준화, 모델 다운로드 및 관리를 가능하게 하는 다용도 사이드카입니다.
* 이기종 GPU 추론: 이기종 GPU를 사용한 비용 효율적인 SLO 기반 LLM 추론입니다.
* GPU 하드웨어 장애 감지: GPU 하드웨어 문제를 사전에 감지합니다.


<CollapsibleContent header={<h2><span>솔루션 배포</span></h2>}>

:::warning
이 블루프린트를 배포하기 전에 GPU 인스턴스 사용과 관련된 비용을 인지하는 것이 중요합니다.
:::

EKS에 AIBrix 모델을 배포하는 방법은 [AI](https://awslabs.github.io/ai-on-eks/docs/infra/aibrix) 페이지를 참조하십시오.

</CollapsibleContent>


### AIBrix 설치 확인

아래 명령을 실행하여 AIBrix 설치를 확인하십시오.

``` bash
kubectl get pods -n aibrix-system
```

모든 Pod가 Running 상태가 될 때까지 기다리십시오.

#### AIBrix 시스템에서 모델 실행

이제 EKS의 AIBrix를 사용하여 Deepseek-Distill-llama-8b 모델을 실행합니다.

아래 명령을 실행하십시오.

```bash
kubectl apply -f blueprints/inference/aibrix/deepseek-distill.yaml
```

이 명령은 deepseek-aibrix 네임스페이스에 모델을 배포합니다. 몇 분 동안 기다린 후 다음을 실행하십시오.

```bash
kubectl get pods -n deepseek-aibrix
```
Pod가 Running 상태가 될 때까지 기다리십시오.

#### 게이트웨이를 사용하여 모델 접근

게이트웨이(Gateway)는 LLM 요청을 처리하도록 설계되었으며 동적 모델 및 LoRA 어댑터 검색, 요청 수 및 토큰 사용량 예산에 대한 사용자 구성, 스트리밍 및 prefix-cache 인식, 이기종 GPU 하드웨어와 같은 고급 라우팅 전략 등의 기능을 제공합니다.
게이트웨이를 사용하여 모델에 접근하려면 아래 명령을 실행하십시오.

```bash
kubectl -n envoy-gateway-system port-forward service/envoy-aibrix-system-aibrix-eg-903790dc 8888:80 &
```

port-forward가 실행되면 게이트웨이에 요청을 전송하여 모델을 테스트할 수 있습니다.

```bash
ENDPOINT="localhost:8888"
curl -v http://${ENDPOINT}/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek-r1-distill-llama-8b",
        "prompt": "San Francisco is a",
        "max_tokens": 128,
        "temperature": 0
    }'
```


<CollapsibleContent header={<h2><span>정리</span></h2>}>

이 스크립트는 `-target` 옵션을 사용하여 모든 리소스가 올바른 순서로 삭제되도록 환경을 정리합니다.

```bash
kubectl delete -f blueprints/inference/aibrix/deepseek-distill.yaml
```

AIBrix 배포를 정리하고 EKS 클러스터를 삭제하려면 아래 명령을 실행하십시오.

```bash
cd infra/aibrix/terraform
./cleanup.sh
```

</CollapsibleContent>

:::caution
AWS 계정에 원치 않는 요금이 부과되지 않도록 이 배포 중에 생성된 모든 AWS 리소스를 삭제하십시오.
:::
