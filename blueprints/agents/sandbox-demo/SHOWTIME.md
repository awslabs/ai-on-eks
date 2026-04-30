# Agent Sandboxes on EKS — Showtime Runbook

Demo-day operational checklist. Not a rewrite of TALK_TRACK.md — that's content, this is pre-flight.

**Target show time**: Thursday April 30, 2026 @ 18:00 local

---

## T-30 — Pre-flight (15 min before showtime, 17:30)

### Cluster health

```bash
# Cluster + nodes alive
kubectl get nodes
# Expected: 2 base nodes + 1-2 gVisor nodes (agent-sandbox-gvisor-*), all Ready.

# Cilium agent
kubectl -n kube-system get pods -l k8s-app=cilium
# Expected: Running on every node

# Hubble UI
kubectl -n kube-system get svc hubble-ui
# Expected: ClusterIP with endpoints
```

### Demo artifacts

```bash
# Both demo Sandbox pods Running
kubectl get pods -n agent-sandboxes
# Expected:
#   sandbox-demo    1/1 Running  (direct-manifest demo)
#   demo-composed   1/1 Running  (KRO-composed demo)

# Policies Valid
kubectl get ciliumclusterwidenetworkpolicies
kubectl get ciliumnetworkpolicies -n agent-sandboxes
# Expected: All VALID True

# KRO RGD
kubectl get rgd
# Expected: agent-sandbox  v1alpha1  AgentSandbox  Active  True
```

### Port-forwards running

Hubble UI port-forward should be in a dedicated terminal window from earlier in the day. If not, start it:

```bash
kubectl port-forward -n kube-system svc/hubble-ui 12000:80
# Leave running in terminal 2 for the whole demo
```

Verify browser tab is open on http://localhost:12000 and shows the Hubble dashboard.

### Agent validation

Quick sanity check the demo agent still runs end-to-end:

```bash
kubectl exec -n agent-sandboxes sandbox-demo -c agent-runtime -- python /workspace/agent.py 2>&1 | tail -10
# Expected tail:
#   Step 1 (PyPI):            PASS — allowed by FQDN policy
#   Step 2 (Bedrock):         PASS — allowed by FQDN policy
#   Step 3 (snippet exec):    PASS — syscalls via Sentry (gVisor)
#   Step 4 (blocked egress):  BLOCKED — denied by FQDN policy
```

If any step fails, debug via the TROUBLESHOOTING section below. **Don't start the demo without this passing.**

### Terminal layout

- **Terminal 1** (demo terminal): `walkthrough.sh run` ready to invoke. cwd = `blueprints/agents/sandbox-demo/`
- **Terminal 2** (Hubble port-forward): Do not touch during demo
- **Terminal 3** (reserve): Available for ad-hoc kubectl if a question needs live investigation

### Browser layout

- Tab 1: Hubble UI (http://localhost:12000), filter set to `namespace=agent-sandboxes`
- Tab 2: TALK_TRACK.md rendered (as cue cards if you want)
- Tab 3: reserve — could pull up the spec / blog post / Mihnea's post for Q&A references

---

## T-0 — Showtime execution

### 18:00 — Opening (1.5 min)

Follow TALK_TRACK.md §0:00-1:30 context. Keep the laptop presenting slides; kubectl is idle.

### 18:01:30 — Architecture (2.5 min)

TALK_TRACK.md §1:30-4:00. Still slides.

### 18:04:00 — Live demo (7 min)

Switch to Terminal 1. Pre-loaded command: `./walkthrough.sh run`

The script walks 6 acts:
1. Show both sandboxes + pods (press ENTER)
2. Confirm gVisor runtime + node (press ENTER)
3. List policies, narrate narrow-scoping lesson (press ENTER)
4. Run the agent — wait for script output ~20s (press ENTER to start)
5. KRO composition walkthrough — RGD + AgentSandbox + children (press ENTER 3 times)
6. Recap via Hubble UI (manual switch to browser)

**Hubble narration** during act 4: point at the green flows for `bedrock-runtime.*` + `pypi.org`, and the DROP flow for `demo-blocked.example.com` when it appears (~20s into agent run).

**Act 5 pacing**: the three KRO commands (rgd, agentsandbox spec, composed children) each get ~30s of narration. Don't rush — this is the composition story most customers will care about.

### 18:11:00 — Roadmap + ask for feedback (2.5 min)

TALK_TRACK.md §11:00-13:30. Four specific feedback prompts at the end — explicitly named so the audience knows when to engage.

### 18:13:30 — Q&A (1.5 min)

TALK_TRACK.md §13:30-15:00 for anticipated questions.

---

## Troubleshooting (if pre-flight fails)

### Sandbox pod not Running

```bash
kubectl describe pod sandbox-demo -n agent-sandboxes | tail -20
```

Common causes:
- **Node not ready** — Karpenter may have consolidated gVisor nodes. Delete + reapply the demo-agent.yaml; Karpenter provisions a fresh one in ~60s.
- **Image pull failure** — docker.io rate limit. Rare but possible; switch `image:` to the ECR mirror `public.ecr.aws/docker/library/python:3.12-slim`.
- **ContainerCreating stuck** — gVisor shim didn't install on the node. Delete the NodeClaim to force Karpenter to reprovision:
  ```bash
  kubectl delete nodeclaim $(kubectl get pod sandbox-demo -n agent-sandboxes -o jsonpath='{.spec.nodeName}' | xargs -I{} kubectl get nodeclaims -o jsonpath='{.items[?(@.status.nodeName=="{}")].metadata.name}')
  ```

### Bedrock call fails during rehearsal

```bash
kubectl exec -n agent-sandboxes sandbox-demo -c agent-runtime -- env | grep AWS
```

Expected: `AWS_WEB_IDENTITY_TOKEN_FILE`, `AWS_ROLE_ARN`, `AWS_REGION` all set.

If not set, the ServiceAccount annotation is missing:

```bash
kubectl annotate serviceaccount sandbox-demo-agent -n agent-sandboxes \
    eks.amazonaws.com/role-arn=arn:aws:iam::893848774378:role/agent-sandbox-sandbox-demo \
    --overwrite
kubectl delete pod sandbox-demo -n agent-sandboxes  # pick up annotation
```

If set but call still fails with AccessDenied, the IAM policy may not cover the model ARN:

```bash
aws iam get-role-policy --role-name agent-sandbox-sandbox-demo --policy-name bedrock-invoke
# Confirm Resource includes arn:aws:bedrock:*::foundation-model/anthropic.claude-sonnet-4-5-*
```

### Hubble UI blank or connection error

Port-forward may have died:

```bash
# Kill any stale port-forwards
pkill -f "port-forward.*hubble-ui"
# Restart
kubectl port-forward -n kube-system svc/hubble-ui 12000:80 &
```

### Cilium not seeing new flows

Cilium policy cache might be stale. Force a reconcile:

```bash
kubectl -n kube-system rollout restart ds/cilium
# Wait ~60s
kubectl -n kube-system rollout status ds/cilium --timeout=2m
```

### Demo blows up mid-run (nuclear option)

Rehearsed fallback: **apologize briefly, switch to slides**, show pre-captured screenshot of a successful Hubble UI flow map + a terminal screenshot of the agent.py PASS/BLOCKED output. Narrate what "would have happened" rather than trying to debug live.

Pre-captured artifacts to have ready:
- Screenshot of Hubble UI showing the green + red flows (capture one before showtime)
- Screenshot of the walkthrough.sh rehearsal terminal output

---

## Post-showtime cleanup (optional, not urgent)

Keep the cluster alive until at least T+1 day in case of follow-up questions or recording re-capture needs. After that:

```bash
cd infra/agent-sandbox/terraform/_LOCAL
./cleanup.sh
```

Destroys VPC, EKS cluster, IAM roles, Karpenter, all AWS-side state. The IRSA role `agent-sandbox-sandbox-demo` + inline Bedrock policy are created separately and need manual cleanup:

```bash
aws iam delete-role-policy --role-name agent-sandbox-sandbox-demo --policy-name bedrock-invoke
aws iam delete-role --role-name agent-sandbox-sandbox-demo
```
