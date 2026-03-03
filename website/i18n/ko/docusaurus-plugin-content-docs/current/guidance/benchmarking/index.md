---
sidebar_label: Amazon EKS에서 LLM 추론 성능 벤치마킹
---

# 벤치마킹 가이드 (Inference Perf 사용)

## 이 가이드에서 다루는 내용

이 가이드는 LLM 추론 성능 벤치마킹에 대한 포괄적인 접근 방식을 제공합니다:

- **[벤치마크 과제 이해하기](./1-understanding-the-benchmark-challenge/index.md)** - LLM 벤치마킹이 복잡한 이유와 기존 AI 모델과의 차이점
- **[LLM 벤치마킹을 위한 핵심 메트릭](./2-key-metrics-for-benchmarking-llms/index.md)** - 필수 메트릭(TTFT, ITL, TPS)과 배포에서의 의미
- **[Inference Perf로 벤치마킹하기](./3-benchmarking-with-inference-perf/1-inference-perf.md)** - 표준화된 Inference Perf 도구를 사용한 성능 측정
- **테스트 시나리오** - 베이스라인, 포화, 프로덕션 시뮬레이션 및 실제 데이터셋 테스트를 위한 실용적인 예제
- **리소스** - 완전한 배포 예제 및 참조 구성
