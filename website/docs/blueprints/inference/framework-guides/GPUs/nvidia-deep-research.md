---
title: NVIDIA Enterprise RAG and AI-Q Research Assistant on EKS
sidebar_position: 9
---

import CollapsibleContent from '@site/src/components/CollapsibleContent';

:::warning
Deployment of Enterprise RAG and AI-Q on EKS requires access to GPU instances (g5, p4, or p5 families). This blueprint relies on [Karpenter](https://karpenter.sh/) autoscaling for dynamic GPU provisioning.
:::

:::info
This blueprint provides two deployment options: **Enterprise RAG Blueprint** (multi-modal document processing with NVIDIA Nemotron and NeMo Retriever Models) or the full **AI-Q Research Assistant** (adds automated research reports with web search). Both run on Amazon EKS with dynamic GPU autoscaling.

Sources: [NVIDIA RAG Blueprint](https://github.com/NVIDIA-AI-Blueprints/rag) | [NVIDIA AI-Q Research Assistant](https://github.com/NVIDIA-AI-Blueprints/aiq-research-assistant)
:::

# NVIDIA Enterprise RAG & AI-Q Research Assistant on Amazon EKS

## What is NVIDIA AI-Q Research Assistant?

[NVIDIA AI-Q Research Assistant](https://build.nvidia.com/nvidia/aiq) is an AI-powered research assistant that creates custom AI researchers capable of operating anywhere, informed by your own data sources, synthesizing hours of research in minutes. The AI-Q NVIDIA Blueprint enables developers to connect AI agents to enterprise data and use reasoning and tools to distill in-depth source materials with efficiency and precision.

### Key Capabilities

**Advanced Research Automation:**
- **5x faster token generation** for rapid report synthesis
- **15x faster data ingestion** with better semantic accuracy
- Summarize diverse data sets with efficiency and precision
- Generate comprehensive research reports automatically

**NVIDIA NeMo Agent Toolkit:**
- Ease development and optimization of agentic workflows
- Unify, evaluate, audit, and debug workflows across different frameworks
- Identify opportunities for optimization
- Flexibly choose and connect agents and tools best suited for each task

**Advanced Semantic Query with NVIDIA NeMo Retriever:**
- Multimodal PDF data extraction and retrieval (text, tables, charts, infographics)
- 15x faster ingestion of enterprise data
- 3x lower retrieval latency
- Multilingual and cross-lingual support
- Reranking to further improve accuracy
- GPU-accelerated index creation and search

**Fast Reasoning with Llama Nemotron:**
- Highest accuracy and lowest latency reasoning capabilities
- Uses [Llama-3.3-Nemotron-Super-49B-v1.5](https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1_5) reasoning model
- Analyze data sources and identify patterns
- Propose solutions based on comprehensive research
- Context-aware generation backed by enterprise data

**Web Search Integration:**
- Real-time web search powered by Tavily API
- Supplements on-premise sources with current information
- Expands research beyond internal documents

### AI-Q Components

Per the [official AI-Q architecture](https://github.com/NVIDIA-AI-Blueprints/aiq-research-assistant):

**1. NVIDIA AI Workbench**
- Simplified development environment for agentic workflows
- Local testing and customization
- Easy configuration of different LLMs
- NVIDIA NeMo Agent Toolkit integration

**2. NVIDIA RAG Blueprint**
- Solution for querying large sets of on-premise multi-modal documents
- Supports text, images, tables, and charts extraction
- Semantic search and retrieval with GPU acceleration
- Foundation for AI-Q's research capabilities

**3. NVIDIA NeMo Retriever Microservices**
- Multi-modal document ingestion
- Graphic elements detection
- Table structure extraction
- PaddleOCR for text recognition
- 15x faster data ingestion

**4. NVIDIA NIM Microservices**
- Optimized inference containers for LLMs and vision models
- [Llama-3.3-Nemotron-Super-49B-v1.5](https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1_5) reasoning model
- Llama-3.3-70B-Instruct model for report generation
- GPU-accelerated inference

**5. Web Search (Tavily)**
- Supplements on-premise sources with real-time web search
- Expands research beyond internal documents
- Powers web-augmented research reports

## What is NVIDIA Enterprise RAG Blueprint?

The [NVIDIA Enterprise RAG Blueprint](https://build.nvidia.com/nvidia/build-an-enterprise-rag-pipeline) is a production-ready reference workflow that provides a complete foundation for building scalable, customizable pipelines for both retrieval and generation. Powered by NVIDIA NeMo Retriever models and NVIDIA Llama Nemotron models, the blueprint is optimized for high accuracy, strong reasoning, and enterprise-scale throughput.

With built-in support for multimodal data ingestion, advanced retrieval, reranking, and reflection techniques, and seamless integration into LLM-powered workflows, it connects language models to enterprise data across text, tables, charts, audio, and infographics from millions of documents—enabling truly context-aware and generative responses.

### Key Features

**Data Ingestion and Processing:**
- **Multimodal PDF data extraction** with text, tables, charts and infographics
- **Audio file ingestion** support
- Custom metadata support
- Document summarization
- Support for millions of documents at enterprise scale

**Vector Database and Retrieval:**
- Multi-collection searchability across document sets
- **Hybrid search** with dense and sparse search
- Reranking to further improve accuracy
- GPU-accelerated index creation and search
- **Pluggable vector database** architecture:
  - ElasticSearch support
  - Milvus support
  - OpenSearch Serverless support (used in this deployment)
- Query decomposition for complex queries
- Dynamic metadata filter generation

**Multimodal and Advanced Generation:**
- Optional **Vision Language Model (VLM)** support in answer generation
- Opt-in image captioning with VLMs
- Multi-turn conversations for interactive Q&A
- Multi-session support for concurrent users
- Improve accuracy with optional reflection

**Governance and Safety:**
- Improve content safety with optional programmable guardrails
- Enterprise-grade security features
- Data privacy and compliance controls

**Observability and Telemetry:**
- Evaluation scripts included (RAGAS framework)
- OpenTelemetry support for distributed tracing
- Zipkin integration for trace visualization
- Grafana dashboards for metrics and monitoring
- Performance profiling and optimization tools

**Developer Features:**
- User interface included for testing and demos
- NIM Operator support for GPU sharing using DRA
- Native Python library support
- OpenAI-compatible APIs for easy integration
- Decomposable and customizable architecture
- Plug-in system for extending functionality

### Enterprise RAG Use Cases

The Enterprise RAG Blueprint can be used standalone or as a component in larger systems:

- **Enterprise search** across document repositories
- **Knowledge assistants** for organizational knowledge bases
- **Generative copilots** for domain-specific applications
- **Vertical AI workflows** customized for specific industries
- **Foundational component** in agentic workflows (like AI-Q Research Assistant)
- **Customer support automation** with context-aware responses
- **Document analysis** and summarization at scale

Whether you're building enterprise search, knowledge assistants, generative copilots, or vertical AI workflows, the NVIDIA AI Blueprint for RAG delivers everything needed to move from prototype to production with confidence. It can be used standalone, combined with other NVIDIA Blueprints, or integrated into an agentic workflow to support more advanced reasoning-driven applications.

## Overview

This blueprint implements the **[NVIDIA AI-Q Research Assistant](https://github.com/NVIDIA-AI-Blueprints/aiq-research-assistant)** on Amazon EKS, combining the [NVIDIA RAG Blueprint](https://github.com/NVIDIA-AI-Blueprints/rag) with AI-Q components for comprehensive research capabilities.

### Deployment Options

This blueprint supports two deployment modes based on your use case:

**Option 1: Enterprise RAG Blueprint**
- Deploy NVIDIA Enterprise RAG Blueprint with multi-modal document processing
- Includes NeMo Retriever microservices and OpenSearch integration
- Best for: Building custom RAG applications, document Q&A systems, knowledge bases

**Option 2: Full AI-Q Research Assistant**
- Includes everything from Option 1 plus AI-Q components
- Adds automated research report generation with web search capabilities via Tavily API
- Best for: Comprehensive research tasks, automated report generation, web-augmented research

Both deployments include [Karpenter](https://karpenter.sh/) autoscaling and enterprise security features. You can start with Option 1 and add AI-Q components later as your needs evolve.

### Deployment Approach

**Why This Setup Process?**
While this implementation involves multiple steps, it provides several advantages:

- **Complete Infrastructure**: Automatically provisions VPC, EKS cluster, OpenSearch Serverless, and monitoring stack
- **Enterprise Features**: Includes security, monitoring, and scalability features
- **AWS Integration**: Leverages [Karpenter](https://karpenter.sh/) autoscaling, EKS Pod Identity authentication, and managed AWS services
- **Reproducible**: Infrastructure as Code ensures consistent deployments across environments

### Key Features

**Performance Optimizations:**
- **[Karpenter](https://karpenter.sh/) Autoscaling**: Dynamic GPU node provisioning based on workload demands
- **Intelligent Instance Selection**: Automatically chooses optimal GPU instance types (G5, P4, P5)
- **Bin-Packing**: Efficient GPU utilization across multiple workloads

**Enterprise Ready:**
- **OpenSearch Serverless**: Managed vector database with automatic scaling
- **Pod Identity Authentication**: EKS Pod Identity for secure AWS IAM access from pods
- **Observability Stack**: Prometheus, Grafana, and DCGM for GPU monitoring
- **Secure Access**: Kubernetes port-forwarding for controlled service access

## Architecture

### AI-Q Research Assistant Architecture

The deployment uses Amazon EKS with [Karpenter](https://karpenter.sh/)-based dynamic provisioning:

![NVIDIA AI-Q on EKS](../../img/nvidia-deep-research-arch.png)


### Enterprise RAG Blueprint Architecture

![RAG Pipeline with OpenSearch](../../img/nvidia-rag-opensearch-arch.png)

The [RAG pipeline](https://github.com/NVIDIA-AI-Blueprints/rag) processes documents through multiple specialized NIM microservices:

**1. Llama-3.3-Nemotron-Super-49B-v1.5**
- [Advanced reasoning model](https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1_5)
- Primary reasoning and generation for both RAG and report writing
- Query rewriting and decomposition
- Filter expression generation

**2. Embedding & Reranking**
- LLama 3.2 NV-EmbedQA: 2048-dim embeddings
- LLama 3.2 NV-RerankQA: Relevance scoring

**3. NV-Ingest Pipeline**
- **PaddleOCR**: Text extraction from images
- **Page Elements**: Document layout understanding
- **Graphic Elements**: Chart and diagram detection
- **Table Structure**: Tabular data extraction

**4. AI-Q Research Assistant Components**
- Llama-3.3-70B-Instruct model for report generation (optional, 2 GPUs)
- Web search via Tavily API
- Backend orchestration for research workflows

## Prerequisites

:::info Important - Cost Information
This deployment uses GPU instances which can incur significant costs. See [Cost Considerations](#cost-considerations) at the end of this guide for detailed cost estimates. **Always clean up resources when not in use.**
:::

**System Requirements**: Any Linux/macOS system with AWS CLI access

Install the following tools:

- **AWS CLI**: Configured with appropriate permissions ([installation guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html))
- **kubectl**: Kubernetes command-line tool ([installation guide](https://kubernetes.io/docs/tasks/tools/install-kubectl/))
- **helm**: Kubernetes package manager ([installation guide](https://helm.sh/docs/intro/install/))
- **terraform**: Infrastructure as code tool ([installation guide](https://learn.hashicorp.com/tutorials/terraform/install-cli))
- **git**: Version control ([installation guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git))

### Required API Tokens

- **NGC API Token**: Required for accessing NVIDIA NIM containers and AI Foundation models
  - **First, sign up through one of these options** (your API key will only work if you have one of these accounts):
    - **Option 1 - NVIDIA Developer Program** (Quick Start):
      - Sign up [here](https://build.nvidia.com/)
      - Free account for POCs and development workloads
      - Ideal for testing and evaluation
    - **Option 2 - NVIDIA AI Enterprise** (Production):
      - Subscribe via [AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-ozgjkov6vq3l6)
      - Enterprise license with full support and SLAs
      - Required for production deployments
  - **Then, generate your API key**:
    - After signing up through Option 1 or 2, generate your API key at [NGC Personal Keys](https://org.ngc.nvidia.com/setup/personal-keys)
    - Keep this key handy - it will be needed at deployment time
- **[Tavily API Key](https://tavily.com/)**: **Optional for AI-Q Research Assistant**
  - Enables web search capabilities in AI-Q
  - AI-Q can work in RAG-only mode without it
  - Not needed for Enterprise RAG only deployment
  - Create account at [Tavily](https://tavily.com/)
  - Generate API key from dashboard
  - Keep this key handy - it will be needed at deployment time if you want web search in AI-Q

### GPU Instance Access

Ensure your AWS account has access to GPU instances. This blueprint supports multiple instance families through [Karpenter](https://karpenter.sh/) NodePools:

**Supported GPU Instance Families:**

| Instance Family | GPU Type | Performance Profile | Use Case |
|----------------|----------|---------------------|----------|
| **G5** (default) | NVIDIA A10G | Cost-effective, 24GB VRAM | General workloads, development |
| **G6e** | NVIDIA L40S | Balanced, 48GB VRAM | High-memory models |
| **P4d/P4de** | NVIDIA A100 | High-performance, 40/80GB VRAM | Large-scale deployments |
| **P5/P5e/P5en** | NVIDIA H100 | Ultra-high performance, 80GB VRAM | Maximum performance |

> **Note**: G5 instances are pre-configured in the Helm values to provide an accessible starting point. You can switch to P4/P5/G6e instances by editing the `nodeSelector` in the Helm values files - no infrastructure changes required.

<CollapsibleContent header={<h4><span>Customizing GPU Instance Types (Optional)</span></h4>}>

:::tip GPU Instance Flexibility
This blueprint is pre-configured with **G5 instances (A10G GPUs)** to provide a cost-effective starting point. However, **you can easily switch to P4 (A100) or P5 (H100) instances** by modifying the Helm values files. The infrastructure includes Karpenter NodePools for G5, G6, G6e, P4, and P5 instance families - simply change the `nodeSelector` labels to match your performance and budget requirements.
:::

All components use Karpenter labels for automatic provisioning. **Default configuration (G5 instances)**:

```yaml
# Example: 8-GPU workloads (49B/70B models)
nodeSelector:
  karpenter.k8s.aws/instance-family: g5  # Use G5 (A10G GPUs)
  karpenter.k8s.aws/instance-size: 48xlarge  # 8x A10G
  karpenter.sh/capacity-type: on-demand

# Example: 1-GPU workloads (embedding, reranking, OCR)
nodeSelector:
  karpenter.k8s.aws/instance-family: g5  # Use G5 (A10G GPUs)
  karpenter.k8s.aws/instance-size: 12xlarge  # Up to 4x A10G
```

**To use different GPU types**, update the `instance-family` in your Helm values:

```yaml
# For P5 (H100 GPUs) - highest performance
nodeSelector:
  karpenter.k8s.aws/instance-family: p5
  karpenter.k8s.aws/instance-size: 48xlarge  # 8x H100

# For P4 (A100 GPUs) - high performance
nodeSelector:
  karpenter.k8s.aws/instance-family: p4d
  karpenter.k8s.aws/instance-size: 24xlarge  # 8x A100

# For G6e (L40S GPUs) - balanced performance
nodeSelector:
  karpenter.k8s.aws/instance-family: g6e
  karpenter.k8s.aws/instance-size: 48xlarge  # 8x L40S
```

**No manual node creation required** - Karpenter automatically provisions the right instances based on your `nodeSelector` configuration!

</CollapsibleContent>

## Getting Started

Clone the repository to begin:

```bash
git clone https://github.com/awslabs/ai-on-eks.git
cd ai-on-eks
```

## Deployment

This blueprint provides two deployment methods:

<CollapsibleContent header={<h2><span>Option A: Automated Deployment (Recommended)</span></h2>}>

Use the provided bash scripts to automate the complete deployment process.

> **💡 Tip**: For detailed manual deployment steps with full configuration control, see [Option B: Manual Deployment](#option-b-manual-deployment) below.

### Step 1: Deploy Infrastructure

Navigate to the infrastructure directory and run the installation script:

```bash
cd infra/nvidia-deep-research
./install.sh
```

This provisions your complete environment:
- **VPC**: Subnets, security groups, NAT gateways
- **EKS Cluster**: With [Karpenter](https://karpenter.sh/) for dynamic GPU provisioning
- **OpenSearch Serverless**: Vector database with Pod Identity authentication
- **Monitoring Stack**: Prometheus, Grafana, and AI/ML observability
- **[Karpenter](https://karpenter.sh/) NodePools**: G5, G6, G6e, P4, P5 instance support

⏱️ **Duration**: 15-20 minutes

> **✅ Infrastructure Ready**: Once Terraform completes successfully, your infrastructure is deployed and ready.

### Step 2: Setup Environment

Run the setup script to configure your environment:

```bash
./deploy.sh setup
```

This script will:
- Configure kubectl to access your EKS cluster
- Collect NGC and Tavily API keys
- Verify cluster readiness (Karpenter, NodePools, OpenSearch)
- Patch Karpenter limits for GPU nodes
- Save configuration to `.env` file

### Step 3: Build OpenSearch Images

Clone RAG source, integrate OpenSearch, and build custom Docker images:

```bash
./deploy.sh build
```

⏱️ **Wait time**: 10-15 minutes for image builds

### Step 4: Deploy Applications

Choose based on your use case:

#### 1) Deploy Enterprise RAG Only

For document Q&A without AI-Q research capabilities:

```bash
./deploy.sh rag
```

⏱️ **Wait time**: 15-25 minutes

**Components deployed:**
- **49B Nemotron Model** (8 GPUs) - [Karpenter](https://karpenter.sh/) will provision g5.48xlarge
- **Embedding & Reranking Models** (1 GPU each)
- **Data Ingestion Models** (1 GPU each)
- **RAG Server** with OpenSearch Serverless integration
- **Frontend** for user interaction

---

#### 2) Deploy AI-Q Research Assistant

AI-Q includes the Enterprise RAG Blueprint plus automated research report generation with optional web search capabilities.

##### Option A: Deploy All at Once (Recommended - Faster)

Deploy both RAG and AI-Q in parallel:

```bash
./deploy.sh all
```

⏱️ **Wait time**: 25-30 minutes

**All components deployed:**
- **RAG**: 49B Nemotron Model, Embedding & Reranking Models, Data Ingestion Models, RAG Server, Frontend
- **AI-Q**: 70B Instruct Model, AIRA Backend, Frontend, Web Search (if Tavily API key provided)

##### Option B: Deploy Sequentially

Deploy RAG first, then add AI-Q:

```bash
# Step 1: Deploy RAG
./deploy.sh rag

# Step 2: Deploy AI-Q
# AI-Q can work with or without web search (Tavily API is optional)
./deploy.sh aira
```

⏱️ **Wait time**: 15-25 minutes for RAG, then 20-30 minutes for AI-Q (35-55 minutes total)


---

</CollapsibleContent>

<CollapsibleContent header={<h2><span>Option B: Manual Deployment</span></h2>}>

Follow detailed manual steps to understand each component and configuration. Ideal for learning, customizing, or troubleshooting.

### Step 1: Deploy Infrastructure

Navigate to the infrastructure directory:

```bash
cd infra/nvidia-deep-research
```

Run the installation script:

```bash
./install.sh
```

⏱️ **Duration**: 15-20 minutes

This provisions:
- VPC with public and private subnets
- EKS Cluster with [Karpenter](https://karpenter.sh/)
- OpenSearch Serverless collection
- Monitoring stack (Prometheus, Grafana)
- [Karpenter](https://karpenter.sh/) NodePools for GPU instances

> **✅ Infrastructure Ready**: Once Terraform completes successfully, your infrastructure is deployed and ready.

### Step 2: Setup Environment

Configure kubectl and set required environment variables:

```bash
# Cluster configuration
export CLUSTER_NAME="nvidia-deep-research"
export REGION="us-west-2"

# Configure kubectl
aws eks update-kubeconfig --region $REGION --name $CLUSTER_NAME

# Verify cluster connection
kubectl get nodes

# Get AWS Account ID
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# OpenSearch Configuration
export OPENSEARCH_SERVICE_ACCOUNT="opensearch-access-sa"
export OPENSEARCH_NAMESPACE="rag"
export COLLECTION_NAME="osv-vector-dev"

# Get OpenSearch endpoint from Terraform output
export OPENSEARCH_ENDPOINT=$(cd terraform/_LOCAL && terraform output -raw opensearch_collection_endpoint)

echo "OpenSearch Endpoint: $OPENSEARCH_ENDPOINT"

# NGC API Key (required)
export NGC_API_KEY="<YOUR_NGC_API_KEY>"

# Tavily API Key for AI-Q (optional - enables web search)
export TAVILY_API_KEY="<YOUR_TAVILY_API_KEY>"  # Skip if deploying RAG only, or for AI-Q without web search
```

### Step 3: Configure [Karpenter](https://karpenter.sh/) NodePool Limits

Increase the memory limit for the G5 GPU NodePool:

```bash
kubectl patch nodepool g5-gpu-karpenter --type='json' -p='[{"op": "replace", "path": "/spec/limits/memory", "value": "2000Gi"}]'
```

This allows [Karpenter](https://karpenter.sh/) to provision sufficient GPU nodes for all models (from 1000Gi to 2000Gi).

### Step 4: Integrate OpenSearch and Build Docker Images

Clone the RAG source code and add OpenSearch implementation:

```bash
# Clone RAG source code
git clone -b v2.3.0 https://github.com/NVIDIA-AI-Blueprints/rag.git rag

# Download OpenSearch implementation
COMMIT_HASH="47cd8b345e5049d49d8beb406372de84bd005abe"
curl -L https://github.com/NVIDIA/nim-deploy/archive/${COMMIT_HASH}.tar.gz | tar xz --strip=5 nim-deploy-${COMMIT_HASH}/cloud-service-providers/aws/blueprints/deep-research-blueprint-eks/opensearch

# Copy OpenSearch implementation into RAG source
cp -r opensearch/vdb/opensearch rag/src/nvidia_rag/utils/vdb/
cp opensearch/main.py rag/src/nvidia_rag/ingestor_server/main.py
cp opensearch/vdb/__init__.py rag/src/nvidia_rag/utils/vdb/__init__.py
cp opensearch/pyproject.toml rag/pyproject.toml

# Login to NGC registry
docker login nvcr.io  # username: $oauthtoken, password: NGC API Key

# Login to ECR
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# Build and push OpenSearch-enabled RAG images to ECR
./opensearch/build-opensearch-images.sh
```

⏱️ **Wait time**: 10-15 minutes for image builds

### Step 5: Deploy Enterprise RAG Blueprint

Deploy the RAG Blueprint using OpenSearch-enabled images:

```bash
# Set deployment variables
export ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
export IMAGE_TAG="2.3.0-opensearch"

# Deploy RAG with OpenSearch configuration
helm upgrade --install rag -n rag \
  https://helm.ngc.nvidia.com/nvidia/blueprint/charts/nvidia-blueprint-rag-v2.3.0.tgz \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --create-namespace \
  --set imagePullSecret.password=$NGC_API_KEY \
  --set ngcApiSecret.password=$NGC_API_KEY \
  --set serviceAccount.create=false \
  --set serviceAccount.name=$OPENSEARCH_SERVICE_ACCOUNT \
  --set image.repository="${ECR_REGISTRY}/nvidia-rag-server" \
  --set image.tag="${IMAGE_TAG}" \
  --set ingestor-server.image.repository="${ECR_REGISTRY}/nvidia-rag-ingestor" \
  --set ingestor-server.image.tag="${IMAGE_TAG}" \
  --set envVars.APP_VECTORSTORE_URL="${OPENSEARCH_ENDPOINT}" \
  --set envVars.APP_VECTORSTORE_AWS_REGION="${REGION}" \
  --set ingestor-server.envVars.APP_VECTORSTORE_URL="${OPENSEARCH_ENDPOINT}" \
  --set ingestor-server.envVars.APP_VECTORSTORE_AWS_REGION="${REGION}" \
  -f helm/rag-values-os.yaml

# Patch ingestor-server to use OpenSearch service account
kubectl patch deployment ingestor-server -n rag \
  -p "{\"spec\":{\"template\":{\"spec\":{\"serviceAccountName\":\"$OPENSEARCH_SERVICE_ACCOUNT\"}}}}"
```

⏱️ **Wait time**: 10-20 minutes for model downloads and GPU provisioning

Verify RAG deployment:

```bash
# Check all pods in RAG namespace
kubectl get all -n rag

# Wait for all pods to be ready
kubectl wait --for=condition=ready pod -l app.kubernetes.io/instance=rag -n rag --timeout=600s

# Verify service accounts
kubectl get pod -n rag -l app.kubernetes.io/component=rag-server -o jsonpath='{.items[0].spec.serviceAccountName}'
kubectl get pod -n rag -l app=ingestor-server -o jsonpath='{.items[0].spec.serviceAccountName}'
```

Deploy DCGM ServiceMonitor for GPU metrics:

```bash
# Deploy ServiceMonitor to connect RAG's Prometheus to infrastructure DCGM Exporter
kubectl apply -f - <<EOF
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: dcgm-exporter
  namespace: rag
  labels:
    release: rag
spec:
  namespaceSelector:
    matchNames:
      - monitoring
  selector:
    matchLabels:
      app.kubernetes.io/name: dcgm-exporter
  endpoints:
    - port: metrics
      interval: 15s
      path: /metrics
EOF
```

This ServiceMonitor allows the Prometheus instance in the `rag` namespace to discover and scrape GPU metrics from the DCGM Exporter running in the `monitoring` namespace.

**Deploy NVIDIA DCGM Grafana Dashboard (Optional but Recommended):**

```bash
# Download and deploy the official NVIDIA DCGM dashboard (with datasource fix)
curl -s https://grafana.com/api/dashboards/12239 | jq -r '.json' | \
    jq 'walk(if type == "object" and has("datasource") and (.datasource | type == "string") then .datasource = {"type": "prometheus", "uid": "prometheus"} else . end)' \
    > /tmp/dcgm-dashboard.json
kubectl create configmap nvidia-dcgm-exporter-dashboard \
    -n rag \
    --from-file=nvidia-dcgm-exporter.json=/tmp/dcgm-dashboard.json \
    --dry-run=client -o yaml | \
    kubectl label --local -f - grafana_dashboard=1 --dry-run=client -o yaml | \
    kubectl apply -f -
```

This dashboard will be automatically loaded by Grafana's sidecar and will display GPU utilization, temperature, memory usage, and other GPU metrics.

---

**Deploy AI-Q Research Assistant (Optional)**

> **📝 Deployment Choice**: Deploy these components if you need automated research report generation with web search capabilities. If your use case only requires the Enterprise RAG Blueprint for document Q&A, proceed to [Access Services](#access-services).

### Step 6: Deploy AI-Q Components

Deploy the AI-Q Research Assistant:

```bash
# Verify TAVILY_API_KEY is set
echo "Tavily API Key: ${TAVILY_API_KEY:0:10}..."

# Deploy AIRA using NGC Helm chart
helm upgrade --install aira https://helm.ngc.nvidia.com/nvidia/blueprint/charts/aiq-aira-v1.2.0.tgz \
  --username='$oauthtoken' \
  --password="${NGC_API_KEY}" \
  -n nv-aira --create-namespace \
  -f helm/aira-values.eks.yaml \
  --set imagePullSecret.password="$NGC_API_KEY" \
  --set ngcApiSecret.password="$NGC_API_KEY" \
  --set tavilyApiSecret.password="$TAVILY_API_KEY"
```

⏱️ **Wait time**: 15-20 minutes for 70B model download

Verify AI-Q deployment:

```bash
# Check all AIRA components
kubectl get all -n nv-aira

# Wait for all components to be ready
kubectl wait --for=condition=ready pod -l app=aira -n nv-aira --timeout=1200s

# Check pod distribution
kubectl get pods -n nv-aira -o wide
```

</CollapsibleContent>

## Access Services

Once deployment is complete, access the services locally using port-forwarding.

<CollapsibleContent header={<h3><span>Port Forwarding Commands</span></h3>}>

**Start Port Forwarding for RAG Services:**

Navigate to the blueprints directory:

```bash
cd ../../blueprints/inference/nvidia-deep-research
```

Start RAG port-forwarding:

```bash
./app.sh port start rag
```

This enables access to:
- **RAG Frontend**: http://localhost:3001 - Test RAG Q&A directly
- **Ingestor API**: http://localhost:8082 - API docs at http://localhost:8082/docs

**Start Port Forwarding for AI-Q Services** (if deployed):

```bash
./app.sh port start aira
```

This enables access to:
- **AIRA Research Assistant**: http://localhost:3000 - Generate comprehensive research reports with web search

**Managing Port Forwarding:**

Check status:
```bash
./app.sh port status
```

Stop port forwarding:
```bash
./app.sh port stop rag      # Stop RAG services
./app.sh port stop aira     # Stop AI-Q services
./app.sh port stop all      # Stop all services
```

</CollapsibleContent>

### Using the Applications

**RAG Frontend (http://localhost:3001):**
- Upload documents directly through the UI
- Ask questions about your ingested documents
- Test multi-turn conversations
- View citations and sources

**AI-Q Research Assistant (http://localhost:3000):**
- Define research topics and questions
- Leverage both uploaded documents and web search
- Generate comprehensive research reports automatically
- Export reports in various formats

**Ingestor API (http://localhost:8082/docs):**
- Programmatic document ingestion
- Batch upload capabilities
- Collection management
- View OpenAPI documentation

## Data Ingestion

After deploying RAG (and optionally AI-Q), you can ingest documents into the OpenSearch vector database.

### Supported File Types

The RAG pipeline supports multi-modal document ingestion including:
- PDF documents
- Text files (.txt, .md)
- Images (.jpg, .png)
- Office documents (.docx, .pptx)
- HTML files

The NeMo Retriever microservices will automatically extract text, tables, charts, and images from these documents.

### Ingestion Methods

You have two options for ingesting documents:

#### Method 1: UI Upload (Testing/Small Datasets)

Upload individual documents directly through the frontend interfaces:

1. **RAG Frontend** (http://localhost:3001) - Ideal for testing individual documents
2. **AIRA Frontend** (http://localhost:3000) - Upload documents for research tasks

This method is perfect for:
- Testing the RAG pipeline
- Small document collections (< 100 documents)
- Quick experimentation
- Ad-hoc document uploads

#### Method 2: S3 Batch Ingestion (Production/Large Datasets)

<CollapsibleContent header={<h4><span>S3 Batch Ingestion Commands</span></h4>}>

Use the data ingestion script to batch process documents from an S3 bucket. Recommended for:
- Production deployments
- Large document collections (hundreds to thousands of documents)
- Automated ingestion workflows
- Scheduled data updates

**Steps:**

1. Ensure the RAG port-forward is running:
   ```bash
   ./app.sh port start rag
   ```

2. Run the data ingestion script (it will prompt for S3 bucket details):
   ```bash
   ./app.sh ingest
   ```

3. Or set environment variables to skip prompts:
   ```bash
   export S3_BUCKET_NAME="your-pdf-bucket-name"
   export S3_PREFIX="documents/"  # Optional folder path
   ./app.sh ingest
   ```

The script will:
- Download documents from your S3 bucket
- Download batch ingestion tools from NVIDIA RAG repository
- Process them through the NeMo Retriever pipeline
- Store embeddings in OpenSearch Serverless
- Display ingestion progress and statistics

> **Additional Resources**:
> - [RAG batch_ingestion.py documentation](https://github.com/NVIDIA-AI-Blueprints/rag/tree/v2.3.0/scripts)
> - [AI-Q bulk data ingestion documentation](https://github.com/NVIDIA-AI-Blueprints/aiq-research-assistant/blob/main/data/readme.md#bulk-upload-via-python)

</CollapsibleContent>

### Verifying Ingestion

After ingestion, verify your documents are available:

1. **Via RAG Frontend**: Navigate to http://localhost:3001 and ask a question about your documents
2. **Via Ingestor API**: Check http://localhost:8082/docs for collection statistics
3. **Via OpenSearch**: Query the OpenSearch collection directly using the AWS Console

## Observability

The RAG and AI-Q deployments include built-in observability tools for monitoring performance, tracing requests, and viewing metrics.

### Access Monitoring Services

**Automated Approach (Recommended):**

Navigate to the blueprints directory and start port-forwarding:

```bash
cd ../../blueprints/inference/nvidia-deep-research
```

```bash
./app.sh port start observability
```

This automatically port-forwards:
- **Zipkin**: http://localhost:9411 - RAG distributed tracing
- **Grafana**: http://localhost:8080 - RAG metrics and dashboards
- **Phoenix**: http://localhost:6006 - AI-Q workflow tracing (if deployed)

Check status:
```bash
./app.sh port status
```

Stop observability port-forwards:
```bash
./app.sh port stop observability
```

<CollapsibleContent header={<h4><span>Manual kubectl Commands</span></h4>}>

**RAG Observability (Zipkin & Grafana):**

```bash
# Port-forward Zipkin for distributed tracing (run in a separate terminal)
kubectl port-forward -n rag svc/rag-zipkin 9411:9411

# Port-forward Grafana for metrics and dashboards (run in another separate terminal)
kubectl port-forward -n rag svc/rag-grafana 8080:80
```

**AI-Q Observability (Phoenix):**

```bash
# Port-forward Phoenix for AI-Q tracing (run in a separate terminal)
kubectl port-forward -n nv-aira svc/aira-phoenix 6006:6006
```

</CollapsibleContent>

### Monitoring UIs

Once port-forwarding is active:

- **Zipkin UI** (RAG tracing): http://localhost:9411
  - View end-to-end request traces
  - Analyze latency bottlenecks
  - Debug multi-service interactions

- **Grafana UI** (RAG metrics): http://localhost:8080
  - Default credentials: admin/admin
  - Pre-built dashboards for RAG metrics
  - GPU utilization and throughput monitoring

- **Phoenix UI** (AI-Q tracing): http://localhost:6006
  - Agent workflow visualization
  - LLM call tracing
  - Research report generation analysis

> **Note**: For detailed information on using these observability tools, refer to:
> - [Viewing Traces in Zipkin](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/observability.md#view-traces-in-zipkin)
> - [Viewing Metrics in Grafana Dashboard](https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/docs/observability.md#view-metrics-in-grafana)

> **Alternative**: If you need to expose monitoring services publicly, you can create an Ingress resource with appropriate authentication and security controls.

## Cleanup

### Uninstall Applications Only

To remove the RAG and AI-Q applications while keeping the infrastructure:

**Using Automation Script (Recommended):**

```bash
cd ../../blueprints/inference/nvidia-deep-research
```

```bash
./app.sh cleanup
```

The cleanup script will:
- Stop all port-forwarding processes
- Uninstall AIRA and RAG Helm releases
- Remove local port-forward PID files

**Manual Application Cleanup:**

```bash
# Navigate to blueprints directory
cd ../../blueprints/inference/nvidia-deep-research

# Stop port-forwards
./app.sh port stop all

# Uninstall AIRA (if deployed)
helm uninstall aira -n nv-aira

# Uninstall RAG
helm uninstall rag -n rag
```

**(Optional) Clean up temporary files created during deployment:**

```bash
rm /tmp/.port-forward-*.pid
```

> **Note**: This only removes the applications. The EKS cluster and infrastructure will remain running. GPU nodes will be terminated by [Karpenter](https://karpenter.sh/) within 5-10 minutes.

### Clean Up Infrastructure

To remove the entire EKS cluster and all infrastructure components:

```bash
# Navigate to infra directory
cd ../../../infra/nvidia-deep-research

# Run cleanup script
./cleanup.sh
```

> **Warning**: This will permanently delete:
> - EKS cluster and all workloads
> - OpenSearch Serverless collection and data
> - VPC and networking resources
> - All associated AWS resources
>
> Backup important data before proceeding.

**Duration**: ~10-15 minutes for complete teardown

## Cost Considerations

<CollapsibleContent header={<h3><span>Estimated Costs for This Deployment</span></h3>}>

:::warning Important
GPU instances and supporting infrastructure can incur significant costs if left running. **Always clean up resources when not in use** to avoid unexpected charges.
:::

### Estimated Monthly Costs

The following table shows approximate costs for the **default deployment** in US West 2 (Oregon) region. Actual costs will vary based on region, usage patterns, and workload duration.

| Resource | Configuration | Estimated Monthly Cost | Notes |
|----------|--------------|----------------------|-------|
| **EKS Control Plane** | 1 cluster | **~$73/month** | Fixed cost: $0.10/hour × 730 hours |
| **GPU Instances (RAG Only)** | 1x g5.48xlarge (8x A10G)<br/>2x g5.12xlarge (4x A10G each) | **~$20,171/month*** | Only when workloads are running<br/>Karpenter scales down when idle |
| **GPU Instances (RAG + AI-Q)** | Additional g5.48xlarge | **~$32,061/month*** | Additional 70B model requires 8 more GPUs |
| **OpenSearch Serverless** | 2-4 OCUs (typical) | **~$350-700/month** | $0.24/OCU-hour<br/>Scales based on data volume and queries |
| **NAT Gateway** | 2 AZs | **~$66/month** | Fixed: 2 gateways × $0.045/hour × 730 hours<br/>Plus data processing: $0.045/GB |
| **ECR Storage** | Docker images | **~$5-10/month** | 50-100GB of custom images<br/>ECR pricing: $0.10/GB/month |
| **EBS Volumes** | Node storage | **~$72/month** | 300GB gp3 per node × 3 nodes × $0.08/GB<br/>Only charged when GPU nodes running |
| **Data Transfer** | Cross-AZ, Internet | **Variable** | Depends on usage patterns<br/>Cross-AZ: $0.01/GB, Internet: $0.09/GB |

**\*GPU Instance Costs assume continuous operation. See breakdown below.**

### GPU Instance Cost Breakdown

GPU instances are the **primary cost driver**. Costs depend on instance type and how long they run:

**Default Configuration (G5 Instances - RAG Only):**

| Instance Type | GPUs | On-Demand Rate | Daily Cost (24hr) | Monthly Cost (730hr) |
|---------------|------|----------------|-------------------|---------------------|
| g5.48xlarge (×1) | 8x A10G | $16.288/hr | $390.91 | $11,890.24 |
| g5.12xlarge (×2) | 4x A10G each | $5.672/hr each | $136.13 each | $4,140.56 each |

**Total for RAG**: ~$20,171/month if running 24/7 (1× g5.48xlarge + 2× g5.12xlarge = $11,890 + $8,281)

**With AI-Q (Additional 70B Model):**
- Additional g5.48xlarge: $11,890.24/month
- **Total**: ~$32,061/month if running 24/7 (2× g5.48xlarge + 2× g5.12xlarge)

> **Note**: If using alternative instance types (G6e, P4, P5), costs will vary. Check [AWS EC2 Pricing](https://aws.amazon.com/ec2/pricing/on-demand/) for your region and instance type.

</CollapsibleContent>

## References

### Official NVIDIA Resources

**📚 Documentation:**
- [NVIDIA AI-Q Research Assistant GitHub](https://github.com/NVIDIA-AI-Blueprints/aiq-research-assistant): Official AI-Q blueprint repository
- [NVIDIA AI-Q on AI Foundation](https://build.nvidia.com/nvidia/aiq): AI-Q blueprint card and hosted version
- [NVIDIA RAG Blueprint](https://github.com/NVIDIA-AI-Blueprints/rag): Complete RAG platform documentation
- [NVIDIA NIM Documentation](https://docs.nvidia.com/nim/): NIM microservices reference
- [NVIDIA AI Enterprise](https://www.nvidia.com/en-us/data-center/products/ai-enterprise/): Enterprise AI platform

**🤖 Models:**
- [Llama-3.3-Nemotron-Super-49B-v1.5](https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1_5): Advanced reasoning model (49B parameters)
- [Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct): Instruction-following model

**📦 Container Images & Helm Charts:**
- [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/): Official container registry
- [RAG Blueprint Helm Chart](https://helm.ngc.nvidia.com/nvidia/blueprint/charts/nvidia-blueprint-rag): Kubernetes deployment
- [NVIDIA NIM Containers](https://catalog.ngc.nvidia.com/orgs/nim): Optimized inference containers

### AI-on-EKS Blueprint Resources

**🏗️ AI-on-EKS Blueprint Resources:**
- [AI-on-EKS Repository](https://github.com/awslabs/ai-on-eks): Main blueprint repository
- [Infrastructure & Deployment Code](https://github.com/awslabs/ai-on-eks/tree/main/infra/nvidia-deep-research): Terraform automation with Karpenter and application deployment scripts
- [Usage Guide](https://github.com/awslabs/ai-on-eks/tree/main/blueprints/inference/nvidia-deep-research): Post-deployment usage, data ingestion, and observability

**📖 Documentation:**
- [Infrastructure & Deployment Guide](https://github.com/awslabs/ai-on-eks/tree/main/infra/nvidia-deep-research/README.md): Step-by-step infrastructure and application deployment
- [Usage Guide](https://github.com/awslabs/ai-on-eks/tree/main/blueprints/inference/nvidia-deep-research/README.md): Accessing services, data ingestion, monitoring
- [OpenSearch Integration](https://github.com/awslabs/ai-on-eks/tree/main/infra/nvidia-deep-research/terraform/opensearch-serverless.tf): Pod Identity authentication setup
- [Karpenter Configuration](https://github.com/awslabs/ai-on-eks/tree/main/infra/nvidia-deep-research/terraform/custom_karpenter.tf): P4/P5 GPU support

### Related Technologies

**☸️ Kubernetes & AWS:**
- [Amazon EKS](https://aws.amazon.com/eks/): Managed Kubernetes service
- [Karpenter](https://karpenter.sh/): Kubernetes node autoscaling
- [OpenSearch Serverless](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/serverless.html): Managed vector database
- [EKS Pod Identity](https://docs.aws.amazon.com/eks/latest/userguide/pod-identities.html): IAM authentication for pods

**🤖 AI/ML Tools:**
- [NVIDIA DCGM](https://developer.nvidia.com/dcgm): GPU monitoring
- [Prometheus](https://prometheus.io/): Metrics collection
- [Grafana](https://grafana.com/): Visualization dashboards

## Next Steps

1. **Explore Features**: Test multi-modal document processing with various file types
2. **Scale Deployments**: Configure multi-region or multi-cluster setups
3. **Integrate Applications**: Connect your applications to the RAG API endpoints
4. **Monitor Performance**: Use Grafana dashboards for ongoing monitoring
5. **Custom Models**: Swap in your own fine-tuned models
6. **Security Hardening**: Add authentication, rate limiting, and disaster recovery

---

This deployment provides the [NVIDIA Enterprise RAG Blueprint](https://github.com/NVIDIA-AI-Blueprints/rag) and [NVIDIA AI-Q Research Assistant](https://github.com/NVIDIA-AI-Blueprints/aiq-research-assistant) on Amazon EKS with enterprise-grade features including [Karpenter](https://karpenter.sh/) automatic scaling, OpenSearch Serverless integration, and seamless AWS service integration.
