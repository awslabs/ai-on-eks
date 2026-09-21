# Lambda MicroVM tier — off-cluster sandbox workers (WIP)

> **Status: pre-review scaffolding.** Part of the agent-sandbox dispatch
> pattern (in-cluster SandboxClaim tiers + this off-cluster tier behind
> one dispatcher). Manifests are static-checked; live validation and the
> dispatcher itself land with the pattern's PR.

The in-cluster tiers (`runc` / `gvisor` / `kata-fc`) select isolation
per-pod via RuntimeClass. This tier selects a different axis entirely:
execution leaves the cluster for AWS Lambda MicroVMs — managed
Firecracker with suspend/resume lifecycle and idle-discount economics —
while staying Kubernetes-native at the control plane through the ACK
`lambdamicrovms` controller (`Microvm` / `MicrovmImage` CRs).

**When this tier**: workload-shape reasons (managed lifecycle,
suspend/resume between turns, no node capacity to manage), not
threat-model reasons. **When not**: FQDN-grained egress (connector-based
only), >8h sessions, x86 or GPU workloads (ARM64-only, 16 vCPU / 32 GiB
cap), fork-to-N snapshot fan-out.

## Pieces

| File | Purpose |
|---|---|
| `manifests/microvm-image-worker.yaml` | Declarative worker image build (`MicrovmImage`): S3 code artifact + managed base image → versioned image |
| `manifests/microvm-worker.yaml` | One worker instance (`Microvm`): image ref by CR name, execution role, idle policy; dispatcher creates these per session and calls `status.endpoint` |

## Prerequisites

1. Base infra with `enable_ack_lambdamicrovms = true` (installs the ACK
   controller + IRSA with the recommended Lambda MicroVM policy)
2. An S3 code artifact (ARM64 worker build) + a build role that can read it
3. A runtime execution role for the worker

## Build path

```
zip worker (arm64) → s3://<bucket>/<key>
  → render __PLACEHOLDERS__ in microvm-image-worker.yaml → kubectl apply
  → controller drives the Lambda image build → CR reaches Available
  → Microvm CRs launch from it via imageIdentifierRef
```
