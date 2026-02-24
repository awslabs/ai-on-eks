---
sidebar_position: 1
sidebar_label: 개요
---

# AI on EKS

[Amazon Elastic Kubernetes Service (EKS)](https://aws.amazon.com/eks/)에서의 AI에 오신 것을 환영합니다. 이 페이지는 대규모 언어 모델(LLM)을 다양한 애플리케이션에 활용하기 위한 출발점입니다. 학습, 파인튜닝, 추론을 위한 아키텍처 패턴과 블루프린트를 탐색해 보세요.

AIoEKS는 블루프린트 배포를 지원하고 AI 분야의 실험 환경이 될 수 있는 조합 가능한 환경을 목표로 합니다.

시작하기에 앞서, 환경에서 사용 가능한 도구들을 살펴보겠습니다:

![OSS ML Platforms on EKS](ml-platforms-eks.png)

## [기반 인프라](https://awslabs.github.io/ai-on-eks/ko/docs/category/infrastructure)
AIoEKS의 핵심은 원하는 환경을 구성하기 위해 조합할 수 있는 모듈 세트입니다. 실험, 학습, 추론 환경을 빠르게 시작할 수 있는 블루프린트를 제공합니다. 앞으로 더 많은 블루프린트를 추가할 예정이며, 원하는 대로 환경을 구성할 수 있습니다.

EKS 클러스터를 배포할 준비가 되셨다면, 인프라 섹션을 확인하세요.

## [학습 (Training)](https://awslabs.github.io/ai-on-eks/ko/docs/category/training-on-eks)
생성형 AI에서의 학습은 방대한 데이터에서 학습하여 인간과 유사한 텍스트를 이해하고 생성하도록 모델을 훈련시키는 과정입니다. 이 과정은 BERT-Large나 Llama2 같은 모델의 기반을 형성하며, 텍스트 요약, 번역 등 다양한 작업을 수행할 수 있게 합니다.

우리 플랫폼은 PyTorch, TensorFlow, TensorRT, vLLM 등 다양한 ML 프레임워크를 지원합니다. JupyterHub를 사용하여 대화형으로 협업하며 모델을 개발할 수 있으며, 데이터 분석, 모델 구축, 실험 실행이 가능합니다. 학습 단계는 Kubeflow 및 Ray와도 통합되어 데이터 전처리부터 학습 및 평가까지 복잡한 머신 러닝 워크플로우를 관리하는 강력한 솔루션을 제공합니다.

LLM의 세계에 뛰어들어 특정 요구사항에 맞는 모델을 학습시킬 준비가 되셨나요? 종합적인 학습 리소스를 확인해 보세요.

## [추론 (Inference)](https://awslabs.github.io/ai-on-eks/ko/docs/category/inference-on-eks)
추론은 학습된 모델을 사용하여 새로운 입력 데이터에 대한 예측이나 출력을 생성하는 과정입니다. 생성형 AI에서 추론은 모델이 텍스트 생성, 번역, 요약 등의 작업을 실시간으로 수행할 수 있게 합니다. 확장 가능한 추론 플랫폼을 구축하는 것은 높은 수요를 처리하고 낮은 지연 시간을 보장하는 데 필수적입니다.

RayServe, NVIDIA Triton Inference Server, KServe 등의 배포 도구를 활용하여 고성능 모델 서빙을 보장하세요. 또한 AWS Neuron(Inferentia용) 및 NVIDIA GPU를 사용한 최적화 기법도 제공합니다.

## 스토리지 및 데이터 관리
효율적인 데이터 스토리지 및 관리는 성공적인 AI/ML 운영의 기본입니다. 우리 플랫폼은 S3, EBS, EFS, FSx 등 AWS 스토리지 솔루션과 통합됩니다. MLflow를 사용하여 모델 레지스트리 및 버전 관리를 수행하고, Amazon ECR로 컨테이너 이미지를 관리할 수 있습니다.

숙련된 실무자든 이 분야에 처음 입문한 분이든, AI on EKS는 언어 모델링의 최신 발전을 활용할 수 있도록 지원합니다. 각 섹션을 탐색하여 여정을 시작하고, Amazon EKS에서 강력한 AI 모델을 구축, 파인튜닝, 배포하는 방법을 알아보세요.
