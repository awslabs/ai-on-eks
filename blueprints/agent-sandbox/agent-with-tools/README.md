# Agent-with-Tools — Sandbox as a Tool Blueprint

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Components](#components)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [How it works](#how-it-works)
- [Adding a new tool](#adding-a-new-tool)
- [Egress tiers](#egress-tiers)
- [Conformance testing](#conformance-testing)
- [Cleanup](#cleanup)
- [Files in this directory](#files-in-this-directory)

## Overview

This blueprint demonstrates the production-common pattern where a **long-lived agent process** invokes **ephemeral sandboxes per tool call** — the inverse of the [reference agent](../) where the agent *is* the sandbox.

In production agent platforms (Modal, E2B, Daytona, Anthropic Code Execution, OpenAI Agents SDK), the agent is a persistent process that spins up isolated execution environments on demand. This blueprint replicates that architecture on EKS using the same agent-sandbox CRDs from the reference blueprint.

**OpenWebUI** provides the user-facing chat interface — a familiar UI for anyone who's worked with Ollama or local LLM deployments. The agent process receives messages from OpenWebUI, reasons with a model (Amazon Bedrock Claude), and invokes sandbox-backed tools when code execution or data analysis is needed.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│ User browser                                                   │
└────────────────┬───────────────────────────────────────────────┘
                 │ HTTPS
┌────────────────┴───────────────────────────────────────────────┐
│ EKS cluster                                                    │
│                                                                │
│  ┌──────────────────┐     ┌────────────────────────────────┐   │
│  │ OpenWebUI        │────▶│ Agent process                  │   │
│  │ (unsandboxed)    │     │ (unsandboxed; long-lived)      │   │
│  │ port 8080        │     │   - calls Bedrock Claude       │   │
│  └──────────────────┘     │   - maps tool calls to sandbox │   │
│                           │     CRDs via kubectl exec      │   │
│                           └──┬─────────────────────────────┘   │
│                              │                                 │
│           ┌──────────────────┼─────────────────────┐           │
│           ▼                  ▼                     ▼           │
│    ┌──────────────┐   ┌──────────────┐    ┌──────────────┐    │
│    │ Sandbox      │   │ Sandbox      │    │ Sandbox      │    │
│    │ (code exec)  │   │ (data        │    │ (future)     │    │
│    │ runtime:     │   │  analysis)   │    │              │    │
│    │ gvisor/runc  │   │ gvisor/runc  │    │              │    │
│    │ ephemeral    │   │ session-     │    │              │    │
│    │              │   │ scoped       │    │              │    │
│    └──────────────┘   └──────────────┘    └──────────────┘    │
└────────────────────────────────────────────────────────────────┘
```

The agent process and OpenWebUI run as normal unsandboxed pods. The sandboxes are the tools the agent invokes — each gets gVisor isolation (Standard EKS) or runc namespace isolation (Auto Mode) and the same FQDN egress controls from the [egress example](../egress/).

## Components

| Component | Role | Image |
|-----------|------|-------|
| OpenWebUI | User-facing chat interface | `ghcr.io/open-webui/open-webui:latest` |
| Agent process | Tool orchestrator — receives chat, calls Bedrock, invokes sandboxes | `python:3.12-slim` (with agent code via ConfigMap) |
| Code-execution sandbox | Ephemeral Python/shell execution environment | `python:3.12-slim` (via SandboxTemplate) |
| Data-analysis sandbox | Sandbox with pandas/numpy/matplotlib for data analysis | `python:3.12-slim` (via SandboxTemplate) |

## Prerequisites

- The [agent-sandbox infrastructure](../../../infra/agent-sandbox/) deployed.
- The [egress example](../egress/) applied (provides FQDN enforcement + Bedrock IRSA role).
- `kubectl` configured against the cluster.
- `BEDROCK_ROLE_ARN` — the IAM role ARN with Bedrock invoke permissions (echoed by `egress/install.sh`).

## Quick Start

```bash
cd blueprints/agent-sandbox/agent-with-tools
./install.sh
```

The installer:
1. Auto-detects compute mode (Standard EKS vs. Auto Mode)
2. Applies sandbox templates for both tools (code-exec + data-analysis)
3. Deploys the agent process with IRSA credentials
4. Deploys OpenWebUI with the agent as its backend
5. Applies egress policies scoped to each component's needs

After install completes, reach the chat UI over a local port-forward. The
OpenWebUI service is `ClusterIP` by design — it is not exposed publicly,
because it proxies Bedrock and can drive code execution in the sandboxes.

```bash
# Forward the UI to your machine
kubectl -n agent-sandboxes port-forward svc/openwebui 8080:8080

# Open http://localhost:8080, create the admin account (auth is on), and chat
```

> Exposing the UI publicly is opt-in. If you need a shared endpoint, keep
> `WEBUI_AUTH=true` and front it with an internal/source-ranged LoadBalancer or
> an authenticated Ingress — don't switch the Service back to a public
> `LoadBalancer`.

## How it works

### Tool-calling loop

1. User sends a message via OpenWebUI
2. OpenWebUI forwards to the agent's OpenAI-compatible API endpoint
3. Agent calls Bedrock Claude with the message + tool definitions
4. If Claude requests a tool call:
   - **code_execute**: Agent creates/reuses a code-execution sandbox, runs the code via `kubectl exec`, returns stdout/stderr
   - **data_analysis_execute**: Agent creates/reuses a data-analysis sandbox (pandas/numpy/matplotlib preinstalled), runs the snippet via `kubectl exec`, returns stdout/stderr
5. Agent returns the final response to OpenWebUI

### Sandbox lifecycle

- **Code-execution sandboxes** are ephemeral — one per tool call. The agent claims a SandboxClaim, executes the code, captures output, and the sandbox can be cleaned up (or reused within a session).
- **Data-analysis sandboxes** reuse the same sandbox pod across tool calls within a session (avoiding repeated cold starts), but each call runs a fresh Python interpreter — variables and imports do **not** persist between calls, so each snippet must be self-contained. (A kernel-backed sandbox that preserves interpreter state across calls is a possible future enhancement.)

Both sandbox types use the same security posture as the reference agent: `readOnlyRootFilesystem`, `runAsNonRoot`, `capabilities.drop: [ALL]`, writable workspace via emptyDir, FQDN egress allowlist.

## Adding a new tool

To add a third tool (e.g., browser automation, shell access):

1. **Create a SandboxTemplate** — copy `manifests/sandbox-code-exec.yaml`, change the image and container setup
2. **Add the tool definition** — edit `agent/tools.py` to add the tool's schema and execution function
3. **Update egress** — if the new tool needs additional FQDN access, add entries to the allowlist manifests

The agent discovers tools from `tools.py` at startup. No other wiring needed.

## Egress tiers

This blueprint uses three egress tiers:

| Component | Tier label | Allowed destinations |
|-----------|-----------|---------------------|
| Agent process | `egress-tier: agent` | Bedrock, STS, OpenWebUI (cluster-internal) |
| Code-exec sandbox | `egress-tier: sandbox` | PyPI, pythonhosted (pip install) |
| Data-analysis sandbox | `egress-tier: sandbox` | PyPI, pythonhosted (pip install) |
| OpenWebUI | `egress-tier: ui` | Agent (cluster-internal only) |

The agent's egress is broader than the sandboxes' — it needs to reach Bedrock. The sandboxes only need package registries for dependency installation. OpenWebUI only talks to the agent (no external egress).

## Conformance testing

```bash
BEDROCK_ROLE_ARN=arn:aws:iam::<account>:role/<role> ./conformance.sh
```

The conformance test validates:
1. User message → agent reasons → code-execution tool invoked → sandbox runs → output returned
2. User message → agent invokes data-analysis tool → sandbox runs snippet → output returned
3. Sandbox egress is restricted (non-allowlisted FQDN blocked)

## Cleanup

```bash
./install.sh uninstall
```

Removes all blueprint-specific resources (agent deployment, OpenWebUI, sandbox templates, configmaps). Does not touch the platform infra or the egress example's policies.

## Files in this directory

| File / subdirectory | Purpose |
|------|---------|
| `install.sh` | Mode-aware installer (deploy/uninstall) |
| `conformance.sh` | End-to-end conformance test |
| `agent/agent_server.py` | Agent process — OpenAI-compatible API server with tool-calling loop |
| `agent/tools.py` | Tool definitions and sandbox execution logic |
| `agent/requirements.txt` | Python dependencies for the agent process |
| `manifests/agent-deployment.yaml` | Agent Deployment + Service |
| `manifests/openwebui-deployment.yaml` | OpenWebUI Deployment + Service + PVC |
| `manifests/sandbox-code-exec-runc.yaml` | Code-exec SandboxTemplate + warm pool (Auto Mode / runc) |
| `manifests/sandbox-code-exec-gvisor.yaml` | Code-exec SandboxTemplate + warm pool (Standard EKS / gVisor) |
| `manifests/sandbox-jupyter-runc.yaml` | Data-analysis SandboxTemplate + warm pool (Auto Mode / runc) |
| `manifests/sandbox-jupyter-gvisor.yaml` | Data-analysis SandboxTemplate + warm pool (Standard EKS / gVisor) |
| `manifests/egress/` | Egress policies for agent + sandbox tiers |

> SandboxClaims are rendered dynamically by `agent/tools.py` at tool-call time (they check out from the warm pools defined in the template files above), so there are no standalone claim manifests.
| `README.md` | This file |
