# Prompt Engineering Tutorial for Mistral on EKS

This blueprint provides a complete prompt engineering tutorial using Ministral 8B deployed on EKS with vLLM.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      EKS Cluster                            │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Namespace: ministral                                  │ │
│  │  ┌──────────────────────────────────────────────────┐  │ │
│  │  │  Pod: ministral-8b                               │  │ │
│  │  │  ┌──────────────────────────────────────────┐    │  │ │
│  │  │  │  vLLM Server                             │    │  │ │
│  │  │  │  - Ministral-8B-Instruct-2410            │    │  │ │
│  │  │  │  - OpenAI-compatible API                 │    │  │ │
│  │  │  │  - Port 8000                             │    │  │ │
│  │  │  └──────────────────────────────────────────┘    │  │ │
│  │  └──────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────┘ │
│                            ▲                                │
│                            │ :8000                          │
│                   Service: ministral-8b                     │
└─────────────────────────────────────────────────────────────┘
                             ▲
                             │ port-forward
                             │
                    ┌────────┴────────┐
                    │  Your Notebook  │
                    │  localhost:8000 │
                    └─────────────────┘
```

## Prerequisites

- AWS CLI configured
- kubectl installed
- Terraform installed
- Hugging Face account with access to Ministral-8B-Instruct-2410

## Quick Start

### Step 1: Create EKS Cluster (if needed)

If you don't have an EKS cluster with GPU support:

```bash
cd ai-on-eks/infra/base/terraform
./install.sh
```

This takes ~20-30 minutes and creates:
- EKS cluster with Karpenter
- GPU NodePools (g5/g6 instances provisioned on-demand)
- Required addons (NVIDIA GPU Operator, etc.)

### Step 2: Configure kubectl

```bash
# Replace <cluster-name> with your cluster name from blueprint.tfvars (default: prompt-eng)
aws eks update-kubeconfig --region us-west-2 --name <cluster-name>
```

### Step 3: Create Hugging Face Token Secret

```bash
# Get your token from https://huggingface.co/settings/tokens
kubectl create namespace ministral
kubectl create secret generic hf-token \
  --from-literal=token=<YOUR_HF_TOKEN> \
  -n ministral
```

### Step 4: Deploy Ministral 8B

```bash
kubectl apply -f ministral-8b-vllm.yaml
```

### Step 5: Wait for Model to Load

```bash
# Watch pod status (takes 3-5 minutes for model download)
kubectl get pods -n ministral -w

# Check logs
kubectl logs -f deployment/ministral-8b -n ministral
```

### Step 6: Port-Forward to Access Locally

```bash
kubectl port-forward service/ministral-8b 8000:8000 -n ministral
```

### Step 7: Test the Endpoint

```bash
# List models
curl http://localhost:8000/v1/models

# Test chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Ministral-8B-Instruct-2410",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Step 8: Run the Notebooks

Open `00_setup.ipynb` and run through the tutorial!

## Notebooks

| Notebook | Topic |
|----------|-------|
| 00_setup.ipynb | Setup and configuration |
| 01_basic_prompt_structure.ipynb | System vs user prompts |
| 02_role_and_purpose.ipynb | Defining assistant identity |
| 03_clear_structured_instructions.ipynb | Organizing complex prompts |
| 04_formatting_and_delimiters.ipynb | Separating data from instructions |
| 05_few_shot_prompting.ipynb | Using examples to guide behavior |
| 06_controlling_output_format.ipynb | Getting structured outputs |
| 07_step_by_step_reasoning.ipynb | Chain-of-thought prompting |
| 08_reducing_hallucinations.ipynb | Grounding responses in facts |
| 09_putting_it_all_together.ipynb | Real-world case studies |

## Cleanup

### Delete the model deployment
```bash
kubectl delete -f ministral-8b-vllm.yaml
```

### Delete the entire EKS cluster (stops all charges)
```bash
cd ai-on-eks/infra/base/terraform
./cleanup.sh
```

## Estimated Costs

| Component | Instance | Cost/Hour |
|-----------|----------|-----------|
| EKS Control Plane | - | $0.10 |
| Base Nodes | m6i.xlarge x2 | $0.38 |
| GPU Node | g5.2xlarge | $1.21 |
| NAT Gateway | - | $0.045 |
| **Total** | | **~$1.75/hr** |

## Troubleshooting

### Pod stuck in Pending
- Karpenter will auto-provision a GPU node - wait 2-3 minutes
- Check Karpenter is provisioning: `kubectl get nodeclaims`
- Check pod events: `kubectl describe pod -n ministral`
- Check Karpenter logs: `kubectl logs -n kube-system -l app.kubernetes.io/name=karpenter`

### Model download slow
- First deployment downloads ~16GB model (3-5 minutes)
- Subsequent deployments use cached image

### Out of memory
- Ministral 8B needs ~16GB GPU RAM
- Works on g5.2xlarge (24GB A10G) or g6e.2xlarge (24GB L4)

### Connection refused
- Ensure port-forward is running
- Check pod is in Running state: `kubectl get pods -n ministral`
- Check service endpoint: `kubectl get endpoints -n ministral`
