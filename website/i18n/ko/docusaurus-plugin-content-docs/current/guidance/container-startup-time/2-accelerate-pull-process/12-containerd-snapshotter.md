---
sidebar_label: containerd snapshotter 사용
---

# containerd snapshotter 사용

## SOCI snapshotter 사용

이 솔루션은 SOCI snapshotter([v0.11.0](https://github.com/awslabs/soci-snapshotter/releases/tag/v0.11.0)+)를 containerd에 플러그인하여 이미지 풀 프로세스를 유기적으로 개선하는 데 단순히 의존합니다. 이것은 현재 EKS AMI의 기본값이 아니지만 결국 기본값이 될 것입니다.

**아키텍처 개요**

아키텍처 변경이 없습니다. 관련 Karpenter 노드 클래스의 `userData` 또는 비 Karpenter 인스턴스 프로비저닝 방식을 위한 시작 템플릿을 통해 워커 노드에 snapshotter를 부트스트랩해야 합니다.

새로운 SOCI snapshotter 구현은 [containerd 2.1.0](https://github.com/containerd/containerd/releases/tag/v2.1.0)에서 도입된 [멀티파트 레이어 페치](https://github.com/containerd/containerd/pull/10177)와 아이디어가 유사하게 큰 레이어를 청크로 풀할 수 있는 비지연 로딩 풀 모드를 도입하여 더 빠르게 풀할 수 있습니다. 인메모리 대신 임시 파일 버퍼를 사용함으로써 SOCI는 레이어 저장 및 압축 해제 작업을 병렬화할 수 있어 훨씬 빠른 이미지 풀이 가능합니다(하드웨어 제한이 허용하는 한).

**구현 가이드**

아래는 위 변경 사항의 개략적인 구현입니다:

:::info
SOCI snapshotter 사용 방법에 대한 완전한 예제는 [이 가이드](https://builder.aws.com/content/30EkTz8DbMjuqW0eHTQduc5uXi6/accelerate-container-startup-time-on-amazon-eks-with-soci-parallel-mode)를 참조하십시오.
:::

```
apiVersion: karpenter.k8s.aws/v1
kind: EC2NodeClass
metadata:
  name: soci-snapshotter
spec:
  role: KarpenterNodeRole-my-cluster
  instanceStorePolicy: RAID0
  subnetSelectorTerms:
    - tags:
        karpenter.sh/discovery: my-cluster-private
  securityGroupSelectorTerms:
    - tags:
        karpenter.sh/discovery: my-cluster-private
  amiSelectorTerms:
    - alias: al2023@latest
  userData: |
    MIME-Version: 1.0
    Content-Type: multipart/mixed; boundary="//"

    --//
    Content-Type: text/x-shellscript; charset="us-ascii"

    # 1. 아키텍처 감지
    # 2. https://github.com/awslabs/soci-snapshotter/releases/download/...에서
    #    v0.11.0+ SOCI snapshotter 버전 다운로드
    # 3. config.toml 파일을 생성하여 snapshotter 구성
    # 4. systemd 구성 파일을 생성하여 snapshotter 서비스 구성
    # 5. snapshotter 활성화

    --//
    Content-Type: application/node.eks.aws

    apiVersion: node.eks.aws/v1alpha1
    kind: NodeConfig
    spec:
      kubelet:
        config:
          imageServiceEndpoint: unix:///run/soci-snapshotter-grpc/soci-snapshotter-grpc.sock
      containerd:
        config: |
          [proxy_plugins.soci]
            type = "snapshot"
            address = "/run/soci-snapshotter-grpc/soci-snapshotter-grpc.sock"
            [proxy_plugins.soci.exports]
              root = "/var/lib/containerd/io.containerd.snapshotter.v1.soci"
          [plugins."io.containerd.grpc.v1.cri".containerd]
            snapshotter = "soci"
            disable_snapshot_annotations = false
            discard_unpacked_layers = false
    --//

```


**주요 이점**

솔루션은 워커 노드의 containerd에 더 성능이 좋은 snapshotter를 플러그인하여 이미지 풀 프로세스를 직접 개선합니다.

**추가 이점**

이 솔루션은 개발 프로세스에 변경이 필요 없고 추가 인프라가 필요 없으며 기본값으로 활성화되면 코드나 구성에 변경이 필요 없습니다.

**트레이드오프**

snapshotter가 기본값이 되기 전에는 위에서 설명한 워커 노드의 `userData` 부트스트래핑을 구현하고 유지해야 합니다. 기본값이 되면 → 없음.
