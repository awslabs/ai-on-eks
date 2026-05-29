# Basic Sandbox — Smallest viable deployment

The smallest viable Sandbox deployment, demonstrating sandboxing alone — no IRSA, no agent script, no Bedrock env vars, no FQDN allowlist. This is the right starting point if you want to add isolation to an existing workload (an inference server, a Jupyter pod, a batch job runner) without buying into the full reference-agent stack.

## What ships

- `sandbox-claim-basic.yaml` — a SandboxClaim that targets one of the basic SandboxTemplates installed by the platform infra.
- `install.sh` — auto-detects the cluster's compute mode (Standard EKS vs Auto Mode), substitutes the right SandboxTemplate name, applies the claim, waits for Ready, and (with `smoke`) runs a smoke test.
- `README.md` — this file.

The basic blueprint defaults to `nginx:alpine` as the workload image, mirroring the canonical Kubernetes [shell-demo example](https://kubernetes.io/docs/tasks/debug/debug-application/get-shell-running-container/). The image is a placeholder — swap it out for your own as described below.

## Prerequisites

- The [agent-sandbox infrastructure](../../../infra/agent-sandbox/) deployed: from a clone of the repo, `cd infra/agent-sandbox && ./install.sh`.
- The basic platform manifests applied (the namespace and the basic SandboxTemplates):
  ```bash
  cd infra/agent-sandbox/manifests
  kubectl apply -f namespace.yaml
  kubectl apply -f sandbox-runc.yaml      # both modes
  # Standard EKS only:
  kubectl apply -f runtimeclass-gvisor.yaml
  kubectl apply -f sandbox-gvisor.yaml
  # See infra/agent-sandbox/README.md for the gVisor Karpenter NodePool.
  ```
- `kubectl` configured against the cluster.

The basic blueprint does not require KRO, Cilium, or any other addon. The minimum tfvars set is:

```hcl
# terraform/blueprint.tfvars
enable_agent_sandbox = true   # SIG-Apps controller + Sandbox CRDs
enable_kro           = false  # Not needed for the basic blueprint
enable_cilium        = false  # Not needed for the basic blueprint
```

## Usage

Apply the claim and wait for the Pod to come up:

```bash
cd blueprints/agent-sandbox/basic
./install.sh
```

Apply + run a smoke test (`nginx -v` inside the sandbox):

```bash
./install.sh smoke
```

Remove the claim:

```bash
./install.sh uninstall
```

The Pod created by the claim is named `sandbox-basic` in the `agent-sandboxes` namespace. Drive it the same way you drive any K8s pod:

```bash
kubectl exec -n agent-sandboxes sandbox-basic -- nginx -v
kubectl exec -n agent-sandboxes sandbox-basic -- /bin/sh -c "ls /var/cache/nginx"
```

## What the basic templates give you

The basic templates (`sandbox-runc` for runc on both modes, `sandbox-gvisor` for gVisor on Standard EKS) ship a hardened Pod spec that any workload can adopt without re-deriving:

- `runAsNonRoot: true`, `runAsUser: 101`, `readOnlyRootFilesystem: true`
- `allowPrivilegeEscalation: false`, `capabilities.drop: [ALL]`
- emptyDir mounts at `/var/cache/nginx`, `/var/run`, `/tmp` (so readOnlyRootFilesystem holds for the default nginx workload)
- Pod labels `egress-tier: sandbox` + `agent-sandbox/tier: <runc|gvisor>` — these are matched by the egress example's network policies if you layer it on later
- `runtimeClassName: gvisor` + matching `tolerations` (gvisor template only) — schedules onto the gVisor Karpenter NodePool

The default workload image is `nginx:alpine` so the templates produce a Pod that runs out of the box. Swap it out by writing your own SandboxTemplate (copy the basic template, change the image, change the volumeMounts to fit your workload, give it a unique `metadata.name`) and pointing a SandboxClaim at it.

## Customizing the workload

Two patterns:

**(1) Write your own SandboxTemplate.** Copy [`sandbox-runc.yaml`](../../../infra/agent-sandbox/manifests/sandbox-runc.yaml) (or the gvisor variant) into your workload manifests, change `metadata.name`, change the container image + ports + volumeMounts, and apply it. Then point a SandboxClaim at the new template name. This is the canonical pattern — the basic templates demonstrate the shape.

**(2) Use one of the agent-shaped templates.** If your workload happens to be a Python agent that wants the Bedrock + ConfigMap + IRSA scaffolding, the [reference agent's templates](../manifests/sandbox-agent-runc.yaml) are ready-made variants. They live in the blueprint, not the platform infra, because they bake in workload-specific assumptions.

The pattern generalizes: every distinct workload-shape gets its own SandboxTemplate, every claim picks the template that matches the workload it's wrapping. This keeps the platform primitives (basic templates) clean and extensible, and keeps workload-specific assumptions (image, env, volumes, IRSA) in the blueprint that introduces them.

## Layering on egress, IRSA, KRO

The basic blueprint is intentionally bare. Once you have it working, you can layer on:

- **FQDN egress enforcement** — see the [egress example](../egress/). Auto-detects Cilium vs ANP based on compute mode; Pod labels on the basic templates already match the example's policy podSelector.
- **IRSA for AWS APIs** — annotate the Pod's ServiceAccount with `eks.amazonaws.com/role-arn=<arn>`. The basic templates use the `default` SA; for IRSA, write a SandboxTemplate with `serviceAccountName: <your-sa>` and create the matching SA + IAM role.
- **KRO composition** — see the [KRO path](../manifests/kro/). Wraps the SandboxClaim + dependencies into a single `AgentSandbox` CRD if your team prefers a higher-level abstraction.

Each layer is independent — a workload can adopt egress without IRSA, IRSA without KRO, KRO without egress, etc. The basic blueprint validates the foundation; everything else is additive.

## Troubleshooting

### `Pod sandbox-basic` stuck in Pending

Most commonly: gVisor tier on Standard EKS is waiting for a Karpenter node to come up. First gVisor node takes 60-90s (Karpenter bootstrap + runsc shim install). Check:

```bash
kubectl describe pod sandbox-basic -n agent-sandboxes
kubectl get nodeclaims -o wide
```

If you're on Auto Mode and seeing scheduling issues, verify the basic runc template applied:

```bash
kubectl get sandboxtemplate -n agent-sandboxes sandbox-runc
```

### `install.sh` fails with "SandboxTemplate missing"

The basic SandboxTemplates haven't been applied yet. Run the prerequisite kubectl commands above, or `infra/agent-sandbox/install.sh` if the cluster isn't up at all.

### Pod runs but you wanted gVisor isolation, got runc

Two possible causes:

- The cluster is in Auto Mode (gVisor isn't available — the claim falls back to `sandbox-runc`).
- On Standard EKS, the gVisor Karpenter NodePool isn't applied. Check `kubectl get nodepool agent-sandbox-gvisor` and apply [`karpenter-nodepool-gvisor.yaml`](../../../infra/agent-sandbox/manifests/karpenter-nodepool-gvisor.yaml) if missing.
