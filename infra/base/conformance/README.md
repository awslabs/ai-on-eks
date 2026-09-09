# Conformance framework (driver + declarative spec)

A reusable end-to-end conformance harness for AI on EKS components. A single
shared **driver** (`run-conformance.sh`) executes a component-supplied
**declarative spec** (`conformance.yaml`): apply manifests → wait for readiness →
run assertions → report PASS/FAIL → clean up. Blueprints and infra components
author a spec; they don't write a bespoke test script.

> Seeds [#334](https://github.com/awslabs/ai-on-eks/issues/334) (generalize the
> cleanup + conformance harnesses for cross-component reuse). The cleanup driver
> lives alongside at `infra/base/cleanup/` and follows the same "shared driver +
> per-component hook" shape.

## Why a driver + spec split

The first conformance harness (`blueprints/agent-sandbox/conformance.sh`) proved
the flow but baked the assertions into bash specific to one workload. Everything
reusable — region/compute-mode resolution, cluster prereqs, pod-readiness waits,
exec-and-assert, result reporting — is component-agnostic. This framework lifts
that spine into a driver so each new component contributes only a declarative
spec, which is faster to author and consistent to run in CI.

## Usage

```bash
infra/base/conformance/run-conformance.sh path/to/conformance.yaml
# Region/cluster auto-resolve (tfvars > AWS_REGION > AWS_DEFAULT_REGION >
# kubectl context > us-west-2). CLUSTER_NAME overrides the cluster.
```

Exit 0 on all-pass, 1 on any failure. No interactive prompts. Requires
`kubectl`, `aws` CLI v2, `jq`, and `yq` (spec parsing).

## Spec schema (`conformance.yaml`)

```yaml
apiVersion: conformance.ai-on-eks/v1
kind: ConformanceSpec
metadata:
  name: <component-name>
spec:
  namespace: <ns>                 # default namespace for resources + assertions

  # Applied in order before assertions; deleted in reverse on cleanup.
  # Paths are relative to the spec file.
  manifests:
    - sample/workload.yaml

  # Readiness gates waited on before assertions run.
  readiness:
    - kind: Pod                   # Pod | Deployment | (extensible)
      name: <name>
      timeout: 300s

  # Declarative assertions, run in order. Each reports PASS/FAIL.
  assertions:
    - name: <id>
      description: <human-readable intent>
      type: podReady              # see "Assertion types" below
      target: <pod-name>

  cleanup:
    deleteManifests: true         # delete spec.manifests in reverse order
```

## Assertion types

| type | params | passes when |
|---|---|---|
| `podReady` | `target` (pod name) | pod reaches `Ready` within the readiness timeout |
| `podExec` | `target`, `container`, `command`, `match` (`contains`\|`equals`\|`regex`), `expected` | `kubectl exec` output matches |
| `nodeExec` | `podRef` (pod whose node to target), `command`, `match`, `expected` | command run on the node (via SSM) matches — for node-level checks like `findmnt --source soci` |
| `jsonpath` | `resource`, `path`, `expected` | `kubectl get -o jsonpath` equals expected |

New assertion types are added in one place — the driver's `run_assertion()`
dispatch — and immediately available to every spec.

## Debugging a failing assertion on the node (`node-debug.sh`)

When a `nodeExec` assertion fails (or you need to understand *why* a runtime
didn't register / a snapshotter didn't engage), inspect the host directly:

```bash
infra/base/conformance/node-debug.sh <node-name>            # default SOCI+kata bundle
infra/base/conformance/node-debug.sh <node-name> "findmnt --source soci"
infra/base/conformance/node-debug.sh --clean                # remove debug pods
```

It runs a long-lived privileged `hostPID` pod in the `default` namespace and
`nsenter`s the host — deliberately **not** SSM. Two reasons, both learned the
hard way during E2E: fresh Karpenter nodes lag SSM registration by minutes
(so `aws ssm send-command` returns `InvalidInstanceId` exactly when you want to
look), and `kube-system` is PodSecurity-restricted (privileged pods are
rejected there; `default` works). The pod tolerates all taints, so it reaches
the gVisor/Kata+FC tiers too.

## Authoring a spec for your component

1. Create `conformance/conformance.yaml` next to your component, plus any sample
   workload manifests it references.
2. Declare the manifests, readiness gates, and assertions.
3. Run the driver against it; iterate until it clears end-to-end on a live
   cluster (spin-up → conformance → cleanup, zero manual intervention).

See `blueprints/agent-sandbox/conformance/` for worked examples (SOCI
image-distribution; Kata+FC isolation tier).
