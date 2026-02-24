---
title: EKS에서의 DeepSeek-R1
sidebar_position: 5
---
import CollapsibleContent from '@site/src/components/CollapsibleContent';

# Ray와 vLLM을 사용한 EKS에서의 DeepSeek-R1

이 가이드에서는 [Amazon EKS](https://aws.amazon.com/eks/)에서 [Ray](https://docs.ray.io/en/latest/serve/getting_started.html)와 [vLLM](https://github.com/vllm-project/vllm) 백엔드를 사용하여 [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) 모델 추론을 배포하는 방법을 살펴봅니다.

![alt text](../../img/dseek0.png)

## GPU 메모리 요구 사항 이해

`DeepSeek-R1-Distill-Llama`와 같은 8B 파라미터 모델을 배포하려면 신중한 메모리 계획이 필요합니다. 각 모델 파라미터는 일반적으로 2바이트(`BF16` 정밀도)를 소비하며, 이는 전체 모델 가중치에 약 `14.99 GiB`의 GPU 메모리가 필요함을 의미합니다. 아래는 배포 중 관찰된 실제 메모리 사용량입니다:

Ray 배포의 로그 샘플

```log
INFO model_runner.py:1115] Loading model weights took 14.99 GiB
INFO worker.py:266] vLLM instance can use total GPU memory (22.30 GiB) x utilization (0.90) = 20.07 GiB
INFO worker.py:266] Model weights: 14.99 GiB | Activation memory: 0.85 GiB | KV Cache: 4.17 GiB
```

G5 인스턴스는 `24 GiB` 메모리의 단일 `A10G` GPU를 제공하여 인스턴스당 하나의 대규모 LLM 추론 프로세스를 실행하는 데 이상적입니다. 이 배포에서는 1x NVIDIA A10G GPU (24 GiB), 16 vCPU 및 64 GiB RAM이 있는 `G5.4xlarge`를 사용합니다.

vLLM을 사용하여 메모리 활용을 최적화하여 OOM(메모리 부족) 충돌을 방지하면서 추론 속도를 최대화합니다.


<CollapsibleContent header={<h2><span>EKS 클러스터 및 애드온 배포</span></h2>}>

기술 스택은 다음을 포함합니다:

- [Amazon EKS](https://aws.amazon.com/eks/) - AWS에서 Kubernetes를 사용하여 컨테이너화된 애플리케이션을 배포, 관리 및 확장하는 것을 단순화하는 관리형 Kubernetes 서비스.

- [Ray](https://docs.ray.io/en/latest/serve/getting_started.html) - 머신러닝 추론 워크로드의 확장 가능하고 효율적인 실행을 가능하게 하는 오픈소스 분산 컴퓨팅 프레임워크.

- [vLLM](https://github.com/vllm-project/vllm) - GPU 실행에 최적화된 대규모 언어 모델(LLM)을 위한 고처리량 및 메모리 효율적인 추론 및 서빙 엔진.

- [Karpenter](https://karpenter.sh/) - G5 인스턴스와 같은 컴퓨팅 리소스를 동적으로 프로비저닝하고 관리하여 애플리케이션 가용성과 클러스터 효율성을 향상시키는 오픈소스 Kubernetes 클러스터 오토스케일러


<a id="사전-요구-사항"></a>
### 사전 요구 사항
시작하기 전에 배포 과정을 원활하게 진행하기 위해 필요한 모든 사전 요구 사항이 갖춰져 있는지 확인하세요. 머신에 다음 도구가 설치되어 있는지 확인하세요:

:::info

데모 과정을 단순화하기 위해 각 블루프린트가 다양한 AWS 서비스를 생성할 수 있어 최소 IAM 역할을 만드는 것이 복잡하므로 관리자 권한이 있는 IAM 역할을 사용한다고 가정합니다. 그러나 프로덕션 배포의 경우 필요한 권한만 있는 IAM 역할을 만드는 것이 강력히 권장됩니다. [IAM Access Analyzer](https://aws.amazon.com/iam/access-analyzer/)와 같은 도구를 사용하면 최소 권한 접근 방식을 보장하는 데 도움이 될 수 있습니다.

:::

1. [aws cli](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
2. [kubectl](https://Kubernetes.io/docs/tasks/tools/)
3. [terraform](https://learn.hashicorp.com/tutorials/terraform/install-cli)
4. [envsubst](https://pypi.org/project/envsubst/)

<a id="배포"></a>
### 배포

저장소 복제

```bash
git clone https://github.com/awslabs/ai-on-eks.git
```

**중요 참고 사항:**

**1단계**: 블루프린트를 배포하기 전에 `blueprint.tfvars` 파일에서 리전을 업데이트하세요.
또한 불일치를 방지하기 위해 로컬 리전 설정이 지정된 리전과 일치하는지 확인하세요.

예를 들어, `export AWS_DEFAULT_REGION="<REGION>"`을 원하는 리전으로 설정하세요:


**2단계**: 설치 스크립트를 실행합니다.

```bash
cd ai-on-eks/infra/jark-stack && chmod +x install.sh
```

```bash
./install.sh
```

<a id="리소스-확인"></a>
### 리소스 확인

설치가 완료되면 Amazon EKS 클러스터를 확인합니다.

EKS로 인증하기 위한 k8s 설정 파일을 생성합니다.

```bash
aws eks --region us-west-2 update-kubeconfig --name jark-stack
```

```bash
kubectl get nodes
```

```text
NAME                                           STATUS   ROLES    AGE    VERSION
ip-100-64-118-130.us-west-2.compute.internal   Ready    <none>   3h9m   v1.30.0-eks-036c24b
ip-100-64-127-174.us-west-2.compute.internal   Ready    <none>   9h     v1.30.0-eks-036c24b
ip-100-64-132-168.us-west-2.compute.internal   Ready    <none>   9h     v1.30.0-eks-036c24b
```

Karpenter 오토스케일러 Nodepool 확인

```bash
kubectl get nodepools
```

```text
NAME                NODECLASS
g5-gpu-karpenter    g5-gpu-karpenter
x86-cpu-karpenter   x86-cpu-karpenter
```

NVIDIA Device 플러그인 확인

```bash
kubectl get pods -n nvidia-device-plugin
```
```text
NAME                                                              READY   STATUS    RESTARTS   AGE
nvidia-device-plugin-gpu-feature-discovery-b4clk                  1/1     Running   0          3h13m
nvidia-device-plugin-node-feature-discovery-master-568b49722ldt   1/1     Running   0          9h
nvidia-device-plugin-node-feature-discovery-worker-clk9b          1/1     Running   0          3h13m
nvidia-device-plugin-node-feature-discovery-worker-cwg28          1/1     Running   0          9h
nvidia-device-plugin-node-feature-discovery-worker-ng52l          1/1     Running   0          9h
nvidia-device-plugin-p56jj                                        1/1     Running   0          3h13m
```

Ray 클러스터를 생성하는 데 사용되는 [Kuberay Operator](https://github.com/ray-project/kuberay) 확인

```bash
kubectl get pods -n kuberay-operator
```

```text
NAME                                READY   STATUS    RESTARTS   AGE
kuberay-operator-7894df98dc-447pm   1/1     Running   0          9h
```

</CollapsibleContent>

## RayServe와 vLLM을 사용한 DeepSeek-R1-Distill-Llama-8B 배포

EKS 클러스터가 배포되고 필요한 모든 구성 요소가 준비되면 `RayServe`와 `vLLM`을 사용하여 `DeepSeek-R1-Distill-Llama-8B`를 배포하는 단계를 진행할 수 있습니다. 이 가이드는 Hugging Face Hub 토큰을 내보내고, Docker 이미지를 생성(필요한 경우)하고, RayServe 클러스터를 배포하는 단계를 설명합니다.

**1단계: Hugging Face Hub 토큰 내보내기**

모델을 배포하기 전에 필요한 모델 파일에 접근하기 위해 Hugging Face로 인증해야 합니다. 다음 단계를 따르세요:

1. Hugging Face 계정을 만듭니다 (없는 경우).
2. 액세스 토큰을 생성합니다:
 - Hugging Face Settings -> Access Tokens로 이동합니다.
 - 읽기 권한이 있는 새 토큰을 만듭니다.
 - 생성된 토큰을 복사합니다.

3. 터미널에서 토큰을 환경 변수로 내보냅니다:

```bash
export HUGGING_FACE_HUB_TOKEN=$(echo -n "Your-Hugging-Face-Hub-Token-Value" | base64)
```

> 참고: 토큰은 Kubernetes 시크릿에서 사용하기 전에 base64로 인코딩되어야 합니다.


**2단계: Docker 이미지 생성**

모델을 효율적으로 배포하려면 Ray, vLLM 및 Hugging Face 의존성을 포함하는 Docker 이미지가 필요합니다. 다음 단계를 따르세요:

- 제공된 Dockerfile을 사용합니다:

```text
gen-ai/inference/vllm-ray-gpu-deepseek/Dockerfile
```

- 이 Dockerfile은 Ray 이미지를 기반으로 하며 vLLM과 Hugging Face 라이브러리를 포함합니다. 이 배포에는 추가 패키지가 필요하지 않습니다.

- Docker 이미지를 빌드하고 Amazon ECR에 푸시합니다

**또는**

- 미리 빌드된 이미지 사용 (PoC 배포용):

사용자 정의 이미지를 빌드하고 푸시하는 것을 건너뛰려면 공개 ECR 이미지를 사용할 수 있습니다:

```public.ecr.aws/data-on-eks/ray-2.41.0-py310-cu118-vllm0.7.0```

> 참고: 사용자 정의 이미지를 사용하는 경우 RayServe YAML 파일의 이미지 참조를 ECR 이미지 URI로 교체하세요.


**3단계: RayServe 클러스터 배포**

RayServe 클러스터는 여러 리소스를 포함하는 YAML 구성 파일에 정의됩니다:
- 배포를 격리하기 위한 네임스페이스.
- Hugging Face Hub 토큰을 안전하게 저장하기 위한 시크릿.
- 서빙 스크립트(OpenAI 호환 API 인터페이스)를 포함하는 ConfigMap.
- 다음을 포함하는 RayServe 정의:
  - x86 노드에 배포된 Ray head 파드.
  - GPU 인스턴스(g5.4xlarge)에 배포된 Ray 워커 파드.

**배포 단계**

> 참고: `ray-vllm-deepseek.yml`의 image: 필드가 사용자 정의 ECR 이미지 URI 또는 기본 공개 ECR 이미지로 올바르게 설정되어 있는지 확인하세요.

RayServe 구성이 포함된 디렉토리로 이동하고 kubectl을 사용하여 구성을 적용합니다

```sh
cd cd ai-on-eks/blueprints/inference/vllm-ray-gpu-deepseek/
envsubst < ray-vllm-deepseek.yml | kubectl apply -f -
```

**출력**

```text
namespace/rayserve-vllm created
secret/hf-token created
configmap/vllm-serve-script created
rayservice.ray.io/vllm created
```

**4단계: 배포 모니터링**

배포를 모니터링하고 파드 상태를 확인하려면 다음을 실행합니다:

```bash
kubectl get pod -n rayserve-vllm
```

:::info

참고: 첫 배포 시 이미지 풀 프로세스에 최대 8분이 걸릴 수 있습니다. 이후 업데이트는 로컬 캐시를 활용합니다. 필요한 의존성만 포함하는 더 가벼운 이미지를 빌드하여 최적화할 수 있습니다.

:::


```text
NAME                                           READY   STATUS            RESTARTS   AGE
vllm-raycluster-7qwlm-head-vkqsc               2/2     Running           0          8m47s
vllm-raycluster-7qwlm-worker-gpu-group-vh2ng   0/1     PodInitializing   0          8m47s
```

이 배포는 또한 여러 포트가 있는 DeepSeek-R1 서비스를 생성합니다:

- `8265` - Ray Dashboard
- `8000` - DeepSeek-R1 모델 엔드포인트


서비스를 확인하려면 다음 명령을 실행합니다:

```bash
kubectl get svc -n rayserve-vllm

NAME             TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)                                         AGE
vllm             ClusterIP   172.20.208.16    <none>        6379/TCP,8265/TCP,10001/TCP,8000/TCP,8080/TCP   48m
vllm-head-svc    ClusterIP   172.20.239.237   <none>        6379/TCP,8265/TCP,10001/TCP,8000/TCP,8080/TCP   37m
vllm-serve-svc   ClusterIP   172.20.196.195   <none>        8000/TCP                                        37m
```

Ray 대시보드에 접근하려면 관련 포트를 로컬 머신으로 포트 포워딩할 수 있습니다:

```bash
kubectl -n rayserve-vllm port-forward svc/vllm 8265:8265
```

그런 다음 [http://localhost:8265](http://localhost:8265)에서 웹 UI에 접근할 수 있으며, Ray 에코시스템 내의 작업 및 액터 배포가 표시됩니다.

:::info

모델 배포에는 약 4분이 걸립니다

:::

![alt text](../../img/dseek1.png)

![alt text](../../img/dseek2.png)

![alt text](../../img/dseek3.png)


## DeepSeek-R1 모델 테스트

이제 DeepSeek-R1-Distill-Llama-8B 채팅 모델을 테스트할 차례입니다.

먼저 kubectl을 사용하여 `vllm-serve-svc` 서비스로 포트 포워드를 실행합니다:

```bash
kubectl -n rayserve-vllm port-forward svc/vllm-serve-svc 8000:8000
```

**테스트 추론 요청 실행:**

```sh
curl -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "messages": [{"role": "user", "content": "Explain about DeepSeek model?"}],
    "stream": false
}'
```

**응답:**

```
{"id":"chatcmpl-b86feed9-1482-4d1c-981d-085651d12813","object":"chat.completion","created":1739001265,"model":"deepseek-ai/DeepSeek-R1-Distill-Llama-8B","choices":[{"index":0,"message":{"role":"assistant","content":"<think>\n\n</think>\n\nDeepSeek는 중국 회사 DeepSeek Inc.에서 개발한 강력한 AI 검색 엔진입니다. 정밀한 추론과 효율적인 계산을 통해 복잡한 STEM(과학, 기술, 공학, 수학) 문제를 해결하도록 설계되었습니다..."},"logprobs":null,"finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":10,"total_tokens":359,"completion_tokens":349,"prompt_tokens_details":null},"prompt_logprobs":null}%
```

## Open Web UI 배포

이제 EKS에 배포된 DeepSeek 모델과 상호 작용하기 위한 ChatGPT 스타일 채팅 인터페이스를 제공하는 오픈소스 Open WebUI를 배포해 보겠습니다. Open WebUI는 모델 서비스를 사용하여 요청을 보내고 응답을 받습니다.

**Open WebUI 배포**

1. Open WebUI용 YAML 파일 `gen-ai/inference/vllm-ray-gpu-deepseek/open-webui.yaml`을 확인합니다. 이것은 EKS에서 컨테이너로 배포되며 모델 서비스와 통신합니다.
2. Open WebUI 배포를 적용합니다:

```bash
cd gen-ai/inference/vllm-ray-gpu-deepseek/
kubectl apply -f open-webui.yaml
```

**출력:**

```text
namespace/openai-webui created
deployment.apps/open-webui created
service/open-webui created
```

**Open WebUI 접근**

웹 UI를 열려면 Open WebUI 서비스를 포트 포워딩합니다:

```bash
kubectl -n open-webui port-forward svc/open-webui 8080:80
```

그런 다음 브라우저를 열고 다음으로 이동합니다: [http://localhost:8080](http://localhost:8080)

등록 페이지가 표시됩니다. 이름, 이메일, 비밀번호로 등록합니다.

![alt text](../../img/dseek4.png)

![alt text](../../img/dseek5.png)

![alt text](../../img/dseek6.png)

![alt text](../../img/dseek7.png)

![alt text](../../img/dseek8.png)

요청을 제출한 후 GPU 및 CPU 사용량이 정상으로 돌아오는 것을 모니터링할 수 있습니다:

![alt text](../../img/dseek9.png)


## 주요 사항

**1. 모델 초기화 및 메모리 할당**
  - 배포되면 모델은 CUDA를 자동으로 감지하고 실행 환경을 초기화합니다.
  - GPU 메모리는 동적으로 할당되며, 모델 가중치(14.99 GiB), 활성화 메모리(0.85 GiB), KV 캐시(4.17 GiB)에 90% 활용률이 예약됩니다.
  - 가중치를 가져오고 추론에 최적화하는 동안 첫 번째 모델 로드 시 약간의 초기 지연이 예상됩니다.

 **2. 추론 실행 및 최적화**
   - 모델은 여러 작업을 지원하지만 기본적으로 텍스트 생성(generate)입니다.
   - Flash Attention이 활성화되어 메모리 오버헤드를 줄이고 추론 속도를 향상시킵니다.
   - CUDA Graph Capture가 적용되어 반복 추론을 더 빠르게 수행할 수 있지만, OOM 문제가 발생하면 gpu_memory_utilization을 줄이거나 eager 실행을 활성화하면 도움이 될 수 있습니다.

 **3. 토큰 생성 및 성능 메트릭**
  - 모델은 입력을 기다리는 동안 처음에 프롬프트 처리량이 0 tokens/sec로 표시됩니다.
  - 추론이 시작되면 토큰 생성 처리량이 ~29 tokens/sec로 안정화됩니다.
  - GPU KV 캐시 활용률은 ~12.5%에서 시작하여 더 많은 토큰이 처리됨에 따라 증가하여 시간이 지남에 따라 더 부드러운 텍스트 생성을 보장합니다.

**4. 시스템 리소스 활용**
  - 병렬 실행을 처리하는 8개의 CPU 및 8개의 CUDA 블록이 예상됩니다.
  - 추론 동시성은 요청당 8192 토큰에 대해 4개의 요청으로 제한되므로 모델이 완전히 활용되면 동시 요청이 대기열에 들어갈 수 있습니다.
  - 메모리 스파이크가 발생하면 max_num_seqs를 낮추면 GPU 부담을 줄이는 데 도움이 됩니다.

**5. 모니터링 및 관측성**
  - 로그에서 평균 프롬프트 처리량, 생성 속도, GPU KV 캐시 사용량을 추적할 수 있습니다.
  - 추론이 느려지면 로그에서 메모리 압력이나 스케줄링 지연을 나타낼 수 있는 보류 중이거나 스왑된 요청을 확인하세요.
  - 실시간 관측성(예: 요청 지연 추적)은 기본적으로 비활성화되어 있지만 더 깊은 모니터링을 위해 활성화할 수 있습니다.

**배포 후 예상 사항**

- 메모리 프로파일링 및 CUDA 그래프 최적화로 인해 모델 초기화에 몇 분이 걸립니다.
- 실행되면 효율적인 메모리 사용과 함께 ~29 tokens/sec의 안정적인 처리량을 볼 수 있습니다.
- 성능이 저하되면 KV 캐시 크기를 조정하거나 메모리 활용률을 낮추거나 안정성 향상을 위해 eager 실행을 활성화하세요.

## 정리

마지막으로 리소스가 더 이상 필요하지 않을 때 정리하고 프로비저닝을 해제하는 방법을 안내합니다.

RayCluster 삭제

```bash
cd ai-on-eks/blueprints/inference/vllm-rayserve-gpu

kubectl delete -f open-webui.yaml

kubectl delete -f ray-vllm-deepseek.yml
```

```bash
cd ai-on-eks/infra/jark-stack/terraform/monitoring

kubectl delete -f serviceMonitor.yaml
kubectl delete -f podMonitor.yaml
```

EKS 클러스터 및 리소스 삭제

```bash
export AWS_DEAFULT_REGION="DEPLOYED_EKS_CLUSTER_REGION>"

cd ai-on-eks/infra/jark-stack/terraform/ && chmod +x cleanup.sh
./cleanup.sh
```
