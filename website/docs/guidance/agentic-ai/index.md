---
sidebar_label: Agentic AI on EKS
---

# Agentic AI on EKS

This section provides guidance for building, deploying, and operating AI agents on Amazon EKS. It is built around the [Agents on EKS](../../infra/agents-on-eks.md) reference environment — an open source environment that brings together source control, CI/CD, observability, vector storage, and MCP tool management into a cohesive infrastructure for running agentic workloads.

## Who Is This For?

Teams looking to move beyond local agent development on their laptops. Whether you're deploying your first agent or building a pipeline to continuously test and promote agent changes, these guides walk through the practical steps using open source tooling on Kubernetes.

## What You'll Learn

- [Best Practices for Agent Development](./best-practices.md) — Patterns for structuring agent code so it transitions smoothly from your laptop to an online environment. Covers dependency management, separating invoke logic for testability, wrapping agents in REST APIs, and the AgentOps philosophy for handling stochastic outputs.

- [Building and Deploying Agents](./building-agents.md) — A focused walkthrough of using the environment: containerizing your agent, pushing code to GitLab, setting up CI/CD to automatically build images, deploying to Kubernetes, and configuring AWS access via Pod Identity.

- [Tracing and Evaluating Agents](./evaluating-agents.md) — Add LangFuse observability to your agent, create evaluation datasets, score responses with LLM-as-a-Judge, and automate the full AgentOps pipeline from build to deploy with quality gates.

- [Agent Tools: Browser and Code Interpreter](./agent-tools.md) — Give your agents the ability to browse the web using Selenium Grid and execute code using Jupyter Enterprise Gateway, both deployed as shared services in the cluster.

## The Environment

The Agents on EKS infrastructure deploys the following components into an EKS cluster:

| Component | Purpose |
|-----------|---------|
| [GitLab](https://about.gitlab.com/) | Source control, container registry, and CI/CD pipelines |
| [LangFuse](https://langfuse.com/) | LLM observability, tracing, and evaluation |
| [Milvus](https://milvus.io/) | Vector database for embeddings and agent memory |
| [MCP Gateway Registry](https://github.com/agentic-community/mcp-gateway-registry) | Discovery and management of MCP servers |
| [Selenium Grid](https://www.selenium.dev/documentation/grid/) | Remote browser automation for agent web browsing |
| [Jupyter Enterprise Gateway](https://jupyter-enterprise-gateway.readthedocs.io/) | Remote kernel management for agent code execution |

For deployment instructions and configuration options, see the [infrastructure guide](../../infra/agents-on-eks.md).
