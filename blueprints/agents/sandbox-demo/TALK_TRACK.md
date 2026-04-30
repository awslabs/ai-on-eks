# Agent Sandboxes on EKS — 15-Minute Showcase Talk Track

**Date**: April 30, 2026
**Event**: Get Ahead, Stay Ahead Showcase
**Presenter**: Brian Hammons (AWS)

Format: 2 min context → 3 min architecture → 5 min live demo → 3 min roadmap → 2 min Q&A.

All commands below are pre-loaded in `walkthrough.sh run` — press ENTER between acts rather than typing live.

---

## Pre-show checklist (15 min before)

- [ ] Cluster + Cilium + agent-sandbox + manifests all Running (`./install.sh` finished cleanly)
- [ ] Demo agent setup complete (`./walkthrough.sh setup` succeeded)
- [ ] gVisor NodePool pre-warmed — a gVisor node exists + the demo Sandbox pod is Ready
- [ ] Hubble UI port-forward running in terminal 2 (`kubectl port-forward -n kube-system svc/hubble-ui 12000:80`)
- [ ] http://localhost:12000 loaded in a browser tab, filter set to `namespace=agent-sandboxes`
- [ ] Terminal 1 is the demo terminal, `walkthrough.sh` cwd
- [ ] One rehearsal run completed — timings confirmed, no surprises

---

## 0:00-2:00 — Context

**Opening beat** — "Last showcase I walked through Firecracker as the hardware-isolated sandbox tier. Today I want to zoom out. Agents on Kubernetes is no longer a design discussion — the community shipped, AWS shipped, customers are building on it right now."

**Three anchor facts**:

1. **March 2026 — `agent-sandbox` became a formal SIG Apps subproject.** Sandbox CRD, warm pools, Python SDK. Released under `agents.x-k8s.io/v1alpha1`. v0.4.3 is current.
2. **April 13, 2026 — Mihnea Spirescu's reference architecture** on AWS Builder Center — first comprehensive EKS reference for this pattern using gVisor + Kata + Firecracker.
3. **Anthropic's Mythos Preview findings** — frontier models can find and exploit sophisticated vulnerabilities at low cost. Anthropic's own defensive recommendations include running agents in kernel-isolated sandboxes and reviewing PRs with security-aware models. We're operationalizing both on EKS.

**The gap I'm filling**: Mihnea's post explicitly leaves three topics for follow-up — **egress filtering**, **credentials**, and **direct Firecracker control plane**. I'm turning those into a shippable blueprint pair for `awslabs/ai-on-eks`.

---

## 2:00-5:00 — Architecture

**Three talking points, slide-backed**:

### The isolation spectrum

```
runc (standard)  →  gVisor  →  Kata + Firecracker
namespace only      userspace     hardware-enforced
~1s cold start      syscall       microVM, ~5s cold
                    ~1.5s cold
```

Key framing — **every tier is a weaker primitive than the one below it**. runc is not a security boundary; namespace isolation + cgroup resource caps is what it is. gVisor's Sentry intercepts syscalls in userspace and only forwards the safe ones to the host kernel. Kata + Firecracker is a real KVM-backed microVM. Tier choice maps to your threat model, not your "is it secure enough?" question.

Today's demo = **gVisor live** (Firecracker already shown previously, standard tier is trivial).

### The egress story — where December 2025 changed things

Before Dec 2025, FQDN egress on EKS required third-party tooling (Cilium, Tetragon, Network Firewall). Since Dec 2025:

- **`ClusterNetworkPolicy`** (admin tier, cluster-wide, EKS Auto + Standard via VPC CNI v1.21.1+)
- **`ApplicationNetworkPolicy`** (namespace-scoped FQDN egress, EKS Auto only today — Standard support "coming weeks")

The customer reality today on **standard EKS** (most real deployments) — you don't have `ApplicationNetworkPolicy` yet. The blueprint's chained Cilium path gives you FQDN egress immediately with a clean migration when AWS extends native support. That's what's running in the cluster behind me.

### What this blueprint delivers

- `infra/agent-sandbox/` — cluster + gVisor + agent-sandbox controller + policies
- `blueprints/agents/sandbox-demo/` — reference Bedrock-backed agent
- Composition layer via kro — optional `AgentSandbox` single-CR abstraction

---

## 5:00-10:00 — Live demo

Drive via `./walkthrough.sh run` in terminal 1, Hubble UI visible in browser.

### Act 1 (0:30) — "The sandbox is running"

```bash
kubectl get sandbox -n agent-sandboxes
kubectl get pods -n agent-sandboxes -o wide
```

**Narration**: "Nothing special about this — it's a pod. What's different is the tier, the policies, the controller lifecycle."

### Act 2 (0:45) — "Confirm it's actually on gVisor"

```bash
kubectl describe pod sandbox-demo -n agent-sandboxes | grep -E "Runtime|Node:"
kubectl get node <node-name> --show-labels
```

**Narration**: "The RuntimeClassName is gvisor. The node has the `agent-sandbox/runtime=gvisor` label, which means Karpenter provisioned it from a dedicated NodePool with AL2023 + the gVisor containerd shim installed via user-data. Standard-tier pods won't land here because of the taint."

### Act 3 (0:45) — "What the policies actually are"

```bash
kubectl get ciliumclusterwidenetworkpolicies
kubectl get ciliumnetworkpolicies -n agent-sandboxes
```

**Narration**: "Two layers. Admin tier blocks IMDS and link-local across every pod on the cluster — that's the defense-in-depth floor. App tier is the per-sandbox FQDN allowlist — this sandbox can reach Bedrock, STS, and PyPI. Nothing else."

### Act 4 (2:30) — "Run the agent" ← the money shot

**Narration setup**: "The agent does four things. First, it pip-installs boto3 — exercises the PyPI allow rule. Second, it calls Bedrock Claude — exercises the Bedrock allow rule. Third, Claude generates a Python snippet and the agent executes it inside the sandbox — exercises gVisor syscall isolation. Fourth, it deliberately tries to egress to a non-allowlisted domain — which should fail."

```bash
kubectl exec -n agent-sandboxes sandbox-demo -c agent-runtime -- python /workspace/agent.py
```

**As it runs**, point to Hubble UI. Expected timeline:
- ~5s in → green flows to pypi.org:443
- ~15s in → green flow to bedrock-runtime.us-east-1.amazonaws.com:443
- ~18s in → snippet executes locally, no egress
- ~20s in → **red DROP flow** to demo-blocked.example.com:443 — policy-denied
- Terminal prints PASS/BLOCKED summary

**Narration at the drop**: "There it is. The policy worked. If a model inside one of these sandboxes gets prompt-injected or generates pathological tool calls, this is where it stops."

### Act 5 (0:30) — "Recap via Hubble"

Show the flow map in Hubble UI filtered to the last minute. Point out the three allowed destinations and the one drop.

**Narration**: "Four flows, three allowed, one dropped. That's the demo — agent sandboxed in gVisor, egress filtered by FQDN, observability via Hubble. Everything you just saw is in the ai-on-eks fork — PR coming in the next few weeks."

---

## 10:00-13:00 — Roadmap

**Phase 1 Production** (next 4-6 weeks):
- Canonicalize what you just saw → merge PR to `awslabs/ai-on-eks`
- Two blueprints: `agents/sandbox/` (this) + `agents/egress-native/` (Auto Mode + native `ApplicationNetworkPolicy`)
- Allowlist template library — LLM APIs, package registries, dev tools, AWS services
- Co-published blog post with Mihnea

**Phase 2** (3-5 weeks after P1):
- Credentials blueprint — LiteLLM proxy sidecar, virtual-key pattern, IAM/Secrets Manager integration. Addresses Mihnea's second follow-up.
- MNG path for gVisor + upstream EKS ticket for Kata+Firecracker MNG support
- Warm pool tuning guide

**Phase 3** (8-12 weeks after P2):
- Agent-aware egress component — complements `ApplicationNetworkPolicy` with per-sandbox scoping, time-bounded allowlists, session-identity tagging. Standalone eBPF DaemonSet.
- Direct Firecracker control plane contribution — closes Mihnea's third follow-up. Upstream contribution to `kubernetes-sigs/agent-sandbox`.

**The kro layer** (across all phases):
- `AgentSandbox` single-CR composition — pick tier + allowlist + IAM role, get Sandbox + ServiceAccount + CiliumNetworkPolicy. Extends upstream's `composing-sandbox-nw-policies/rgd.yaml`.
- Production shape for this pattern — give customers the one-CR experience without forcing them into operator development.

---

## 13:00-15:00 — Q&A

**Anticipated questions**:

**Q**: Why gVisor not Kata+Firecracker?
**A**: Both are supported tiers. Showed Firecracker last time, today I'm showing gVisor. Customers pick tier based on threat model — gVisor is right when you need syscall-level isolation without the microVM overhead. The Kata tier is in the spec, blocked today on EKS MNG CPU Options support for nested virt — documented limitation, self-managed path works now.

**Q**: Why chained Cilium instead of native `ApplicationNetworkPolicy`?
**A**: `ApplicationNetworkPolicy` is Auto-Mode-only today. Standard EKS support is committed "in the coming weeks" by AWS. Until then, chained Cilium is the path for standard-EKS customers, which is most real deployments. Clean migration when AWS ships.

**Q**: What about credential injection?
**A**: Phase 2 deliverable. Any secret visible inside the sandbox is accessible to a prompt-injected agent — Mihnea flagged this explicitly. The Phase 2 design is a LiteLLM proxy sidecar with virtual keys + langfuse trace correlation. Separate blueprint so teams that only need sandboxing aren't forced into proxy architecture.

**Q**: Can I run this outside AWS?
**A**: The sandbox layer (agent-sandbox + gVisor + Cilium) is portable. The AWS-specific bits are VPC CNI (swap for Cilium-native networking or another CNI), Bedrock (swap for any LLM endpoint), and Karpenter (swap for Cluster Autoscaler). The blueprint is AWS-opinionated for the ai-on-eks repo; the pattern is portable.

**Q**: What about agents that need to actually talk to the internet?
**A**: They go in the allowlist. The blueprint ships four template CiliumNetworkPolicies — LLM APIs, package registries, dev tools, AWS services. You compose the allowlist you need. For genuinely uncontrolled egress, this isn't the right architecture — go back to a VPN or egress gateway.

**Q**: How does this compose with langfuse / evaluation?
**A**: Good question — open question in the spec today. langfuse-based evaluation lives in the existing ai-on-eks evaluating-agents guide. Phase 2's credential proxy integrates with langfuse for trace correlation (sandbox session → proxy request → upstream LLM call). The two patterns layer rather than conflict.

---

## Fallback plays (if live demo has issues)

- **Cluster unreachable** — pivot to walkthrough.sh output from rehearsal (have it screenshot-ready).
- **Sandbox pod crashlooping** — `kubectl describe` it, explain what the node labels show; skip to the policy discussion.
- **Bedrock call fails** — explain the IRSA setup, show the network policy, demonstrate the BLOCKED path against a non-allowlisted domain (that still works even without Bedrock).
- **Hubble UI dead** — show the policy YAML + Cilium CLI output directly (`kubectl exec ds/cilium -- cilium monitor`).

---

## Talking-point cheat sheet (for screen-edge cue card)

- v0.4.3 is the current agent-sandbox release
- VPC CNI v1.21.1+ = FQDN-capable
- gVisor = userspace Sentry, syscall interception
- Firecracker = hardware KVM microVM (shown previously)
- Chained Cilium today → native `ApplicationNetworkPolicy` tomorrow
- Phase 1 PR target = `awslabs/ai-on-eks`
- Co-author = Mihnea Spirescu
- Upstream RGD = `composing-sandbox-nw-policies/rgd.yaml`
