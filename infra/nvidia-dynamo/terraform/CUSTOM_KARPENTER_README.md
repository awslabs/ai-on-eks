# Custom Karpenter Node Pools for NVIDIA Dynamo

This module provides custom Karpenter node pools with higher weights than the base addons, specifically designed for NVIDIA Dynamo workloads including BuildKit support.

## Features

- **Higher Priority**: Weight 100 vs base addons weight 50 for Dynamo workload priority
- **Dual AMI Support**: AL2023 (default) or Bottlerocket with user namespace configuration
- **BuildKit Compatible**: Includes user namespace configuration for rootless BuildKit
- **Optimized Instance Types**: C7i for CPU workloads, G6 for GPU workloads
- **Flexible Configuration**: Easy switching between AMI families

## Node Pools

### 1. C7i CPU Node Pool
- **Instance Family**: c7i (latest generation compute optimized)
- **Instance Sizes**: large to 48xlarge
- **Use Cases**: CPU-intensive workloads, BuildKit, general compute
- **Labels**:
  - `dynamo.ai/node-type: c7i-cpu`
  - `dynamo.ai/buildkit-compatible: true`

### 2. G6 GPU Node Pool
- **Instance Family**: g6 (NVIDIA L4 GPUs)
- **Instance Sizes**: large to 48xlarge
- **Use Cases**: GPU workloads, ML inference, training
- **Labels**:
  - `dynamo.ai/node-type: g6-gpu`
  - `dynamo.ai/buildkit-compatible: true`
  - `accelerator: nvidia`
  - `gpuType: l4`
- **Taints**: `nvidia.com/gpu=true:NoSchedule`

## Configuration

### Variables

```hcl
# Enable custom node pools
enable_custom_karpenter_nodepools = true

# Choose AMI family (default: AL2023)
use_bottlerocket = false  # true for Bottlerocket with user namespaces

# Resource limits
karpenter_cpu_limits = "2000"
karpenter_memory_limits = "2000"
```

### AMI Family Options

#### AL2023 (Default - Recommended)
- **Pros**: Better compatibility, user namespaces enabled by default
- **Cons**: None significant
- **Best for**: Most workloads, BuildKit, general use

#### Bottlerocket (Optional)
- **Pros**: Minimal attack surface, immutable OS
- **Cons**: More complex configuration
- **Best for**: Security-focused deployments
- **Includes**: Custom user namespace configuration for BuildKit

## BuildKit Support

Both AMI families support rootless BuildKit:

- **AL2023**: User namespaces enabled by default in kernel
- **Bottlerocket**: Custom configuration sets `user.max_user_namespaces = 16384`

## Usage

1. **Configuration is in blueprint.tfvars**:
```hcl
# Custom Karpenter Node Pools Configuration
enable_custom_karpenter_nodepools = true
use_bottlerocket = false  # or true for Bottlerocket

# Resource limits (adjust based on workload needs)
karpenter_cpu_limits = 10000
karpenter_memory_limits = 10000
```

2. **Deploy with install.sh**:
```bash
cd infra/nvidia-dynamo
./install.sh
```

3. **Verify deployment**:
```bash
kubectl get nodepools
kubectl get ec2nodeclasses
```

## Node Selection

To schedule pods on these custom nodes, use node selectors:

```yaml
# For C7i CPU nodes
nodeSelector:
  dynamo.ai/node-type: c7i-cpu
  dynamo.ai/buildkit-compatible: "true"

# For G6 GPU nodes
nodeSelector:
  dynamo.ai/node-type: g6-gpu
  dynamo.ai/buildkit-compatible: "true"
```

## Troubleshooting

### BuildKit Issues
If BuildKit fails with user namespace errors:
1. Check if using Bottlerocket: `kubectl get nodes -o wide`
2. Verify user namespace config: `kubectl describe node <node-name>`
3. Consider switching to AL2023: `use_bottlerocket = false`

### Node Scheduling
If pods don't schedule on custom nodes:
1. Check node pool weights: `kubectl describe nodepool`
2. Verify node selectors and taints
3. Check resource requests vs limits

## Integration

These custom node pools integrate with:
- Base EKS module (inherits IAM roles, subnets, security groups)
- Existing Karpenter installation
- NVIDIA device plugins
- BuildKit and container builds

## Files

- `custom-karpenter.tf`: Main node pool definitions
- `custom-karpenter-variables.tf`: Configuration variables and data sources
- `custom-karpenter-outputs.tf`: Output values
- `blueprint.tfvars`: Main configuration file (includes custom Karpenter settings)
