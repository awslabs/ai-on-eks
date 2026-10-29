# Lambda MicroVM tier — off-cluster sandbox workers

> **Status: validated end to end** (Standard EKS rig, us-west-2,
> Sep 2026). Part of the agent-sandbox dispatch pattern (in-cluster
> SandboxClaim tiers + this off-cluster tier behind one dispatcher):
> image build via `MicrovmImage` CR, per-session `Microvm` bind through
> the dispatcher (~32s create → RUNNING + endpoint), worker round-trip
> through the Lambda proxy, label-selector release with zero orphans.

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
   controller + IRSA with the recommended Lambda MicroVM policy,
   `iam:PassRole`, and `lambda:PassNetworkConnector`)
2. An S3 code artifact + a build role that can read it. The artifact is
   a zip of **app code + a Dockerfile** — Lambda runs the Dockerfile
   during the build, starts `CMD`, and snapshots the running state
   (S3 zip only; there is no ECR image path). Both the build and
   execution roles must trust `lambda.amazonaws.com` for
   `sts:AssumeRole` **and** `sts:TagSession`.
3. A runtime execution role for the worker

## Build path

```
zip (code + Dockerfile) → s3://<bucket>/<key>
  → render __PLACEHOLDERS__ in microvm-image-worker.yaml → kubectl apply
  → controller drives the Lambda image build → CR reaches CREATED
  → Microvm CRs launch from it via imageIdentifierRef
```

## Reaching the worker

Every request through the VM's HTTPS endpoint needs a port-scoped auth
token (`create-microvm-auth-token`) in the `X-aws-proxy-auth` header.
The proxy routes to port **8080 by default**; our workers serve on
:8888, so requests also carry `X-aws-proxy-port: 8888` (the port must
be within the token's `allowedPorts`). See `../dispatch/tests/microvm_smoke.py`
for the full live sequence.
