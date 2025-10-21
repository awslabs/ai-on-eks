# Hello World Example

A simple CPU-only example that demonstrates basic NVIDIA Dynamo functionality without requiring GPUs. Perfect for testing your Dynamo platform deployment and understanding core concepts.

## Architecture

```text
Frontend (client.py) → Hello World Worker (hello_world.py)
```

This example demonstrates:
- **Basic Service Discovery**: Frontend discovers worker via etcd
- **Simple Backend Service**: Worker creates a `hello_world/backend` service with a `generate` endpoint
- **Client Communication**: Frontend connects to backend and processes streaming responses
- **CPU-Only Deployment**: No GPU requirements for testing infrastructure

## Key Features

### Learning Objectives
- **Service Discovery**: Understand how Dynamo components find each other
- **Basic Communication**: See worker-frontend communication patterns
- **Resource Management**: Learn CPU-only node selection
- **Health Checking**: Observe probe configuration for different component types
- **Container Structure**: Explore NGC container file system layout

### Educational Value
- **Minimal Complexity**: No model loading or GPU dependencies
- **Fast Deployment**: Quick startup for testing platform functionality
- **Clear Logs**: Easy to understand output for debugging
- **Foundation**: Base pattern for more complex examples

## Prerequisites

- Dynamo platform deployed in your EKS cluster
- `dynamo-cloud` namespace configured
- CPU nodes available (no GPU requirements)

## YAML Structure Explained

### Frontend Configuration
```yaml
Frontend:
  dynamoNamespace: hello-world     # Service discovery namespace
  componentType: main              # Marks as entry point
  replicas: 1                      # Single client instance
  resources:
    requests:
      cpu: "1"                     # Minimal CPU requirements
      memory: "2Gi"                # Basic memory allocation
  extraPodSpec:
    nodeSelector:
      karpenter.sh/nodepool: cpu-karpenter  # CPU-only nodes
    mainContainer:
      image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.5.0
      workingDir: /workspace/examples/runtime/hello_world/
      args: ["python3", "client.py"]
  livenessProbe:
    httpGet:
      path: /health
      port: 8000
    # Note: Hello world doesn't actually serve HTTP, so this will fail
  readinessProbe:
    exec:
      command: ["echo", "ok"]      # Simple successful probe
```

**Key Points:**
- **Client Pattern**: Frontend runs `client.py` instead of HTTP server
- **Exec Probes**: Uses command execution instead of HTTP for readiness
- **CPU Nodes**: Scheduled on cost-effective CPU-only instances
- **Minimal Resources**: Demonstrates lowest resource footprint

### Worker Configuration
```yaml
HelloWorldWorker:
  dynamoNamespace: hello-world     # Must match frontend namespace
  componentType: worker            # Processing unit
  replicas: 1                      # Single worker instance
  resources:
    requests:
      cpu: "1"                     # Minimal CPU for simple service
      memory: "4Gi"                # Memory for service framework
  extraPodSpec:
    nodeSelector:
      karpenter.sh/nodepool: cpu-karpenter  # CPU-only nodes
    mainContainer:
      image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.5.0
      workingDir: /workspace/examples/runtime/hello_world/
      args: ["python3", "hello_world.py", "2>&1", "|", "tee", "/tmp/hello_world.log"]
  readinessProbe:
    exec:
      command: ["/bin/sh", "-c", 'grep "Serving endpoint" /tmp/hello_world.log']
    # Waits for service to register and start serving
```

**Key Parameters:**
- **Service Registration**: Worker registers with Dynamo service discovery
- **Log-Based Probes**: Readiness determined by log file content
- **CPU Scheduling**: Both components run on CPU nodes
- **Output Logging**: Captures output to `/tmp/hello_world.log`

## Node Selection Strategy

### CPU-Only Deployment
```yaml
extraPodSpec:
  nodeSelector:
    karpenter.sh/nodepool: cpu-karpenter  # Target CPU node pool
```

**Why CPU Nodes for Hello World:**
- **No GPU Required**: Pure computational example with no inference
- **Cost Effective**: CPU instances are much cheaper than GPU instances
- **Fast Provisioning**: CPU nodes start faster than GPU nodes
- **Broad Availability**: CPU instances available in all regions/AZs

### Alternative Node Configurations

**For Testing on Existing GPU Nodes:**
```yaml
nodeSelector:
  karpenter.sh/nodepool: g5-gpu-karpenter  # Will work but wasteful
tolerations:
- key: nvidia.com/gpu
  operator: Exists
  effect: NoSchedule
```

**For Specific Instance Types:**
```yaml
nodeSelector:
  karpenter.sh/nodepool: cpu-karpenter
  node.kubernetes.io/instance-type: c5.large  # Specific CPU instance
```

## Deployment

Deploy using kubectl:

```bash
kubectl apply -f hello-world.yaml -n dynamo-cloud
```

Or use the main deployment script:

```bash
cd ..
./deploy.sh hello-world
```

## Testing

### Basic Deployment Verification
```bash
# Check that pods are created
kubectl get pods -n dynamo-cloud -l app=hello-world

# Monitor pod startup
kubectl get pods -n dynamo-cloud -l app=hello-world -w
```

### Service Discovery Testing
```bash
# Check worker registration and service startup
kubectl logs -l componentType=worker,app=hello-world -n dynamo-cloud

# Expected worker output:
# Service registered: hello_world/backend
# Serving endpoint: generate
# Worker ready and waiting for requests
```

### Client Communication Testing
```bash
# Check client execution and output
kubectl logs -l componentType=main,app=hello-world -n dynamo-cloud

# Expected client output:
# Connected to hello_world/backend service
# Received: Hello world!
# Received: Hello sun!
# Received: Hello moon!
# Received: Hello star!
# Client completed successfully
```

### Understanding the Workflow
1. **Worker Startup**: `hello_world.py` starts and registers service with etcd
2. **Service Discovery**: `client.py` discovers the worker service
3. **Communication**: Client connects and requests streaming data
4. **Completion**: Client prints responses and exits (this is expected behavior)

### Pod Lifecycle
- **Worker Pod**: Should remain `Running` (long-lived service)
- **Frontend Pod**: May show `Completed` after client finishes (this is normal)

## Advanced Testing

### Exploring Container Structure
```bash
# List available examples in the container
kubectl exec -it <worker-pod> -n dynamo-cloud -- ls -la /workspace/examples/

# Examine the hello world source code
kubectl exec -it <worker-pod> -n dynamo-cloud -- cat /workspace/examples/runtime/hello_world/hello_world.py

# Check Dynamo runtime libraries
kubectl exec -it <worker-pod> -n dynamo-cloud -- find /workspace -name "*.py" | grep dynamo | head -10
```

### Service Discovery Debugging
```bash
# Check etcd for service registration (if accessible)
kubectl exec -it <worker-pod> -n dynamo-cloud -- python3 -c "
import os
from dynamo.runtime import service
print('Services registered:', service.list_services())
"
```

## Monitoring

### Pod Status
```bash
# Check all hello-world components
kubectl get pods -n dynamo-cloud -l app=hello-world -o wide

# Check resource usage
kubectl top pods -n dynamo-cloud -l app=hello-world
```

### Logs and Events
```bash
# Watch all hello-world logs
kubectl logs -n dynamo-cloud -l app=hello-world -f --all-containers

# Check pod events for any issues
kubectl get events -n dynamo-cloud --sort-by='.lastTimestamp' | grep hello-world
```

### DynamoGraphDeployment Status
```bash
# Check deployment status
kubectl get dynamographdeployment hello-world -n dynamo-cloud -o yaml

# Monitor deployment progress
kubectl describe dynamographdeployment hello-world -n dynamo-cloud
```

## Troubleshooting

### Common Issues

**Worker Pod Not Starting:**
```bash
# Check pod events and logs
kubectl describe pod <worker-pod> -n dynamo-cloud
kubectl logs <worker-pod> -n dynamo-cloud

# Common causes:
# - Image pull issues
# - Node scheduling problems
# - Resource constraints
```

**Client Not Connecting to Worker:**
```bash
# Verify worker service registration
kubectl logs <worker-pod> -n dynamo-cloud | grep -i "service\|register\|endpoint"

# Check client discovery attempts
kubectl logs <frontend-pod> -n dynamo-cloud | grep -i "discover\|connect"

# Common causes:
# - Namespace mismatch between frontend and worker
# - etcd connectivity issues
# - Worker not fully initialized
```

**Frontend Pod Stuck in Pending:**
```bash
# Check node availability
kubectl get nodes -l karpenter.sh/nodepool=cpu-karpenter

# Verify CPU node provisioning
kubectl describe pod <frontend-pod> -n dynamo-cloud | grep -i pending

# Common causes:
# - No CPU nodes available
# - Resource requests too high
# - Node selector issues
```

### Performance Notes
- **Startup Time**: Should complete in under 2 minutes
- **Resource Usage**: Minimal CPU/memory consumption
- **Network**: All communication via internal cluster networking

## External Access

For production-grade external access to your hello-world service:

### Option 1: Kubernetes Service + AWS Load Balancer
Create a Service and use AWS Load Balancer Controller:

```bash
# Create a Service for the frontend
kubectl expose deployment hello-world-frontend --port=8000 --target-port=8000 --type=LoadBalancer -n dynamo-cloud

# Or use AWS Load Balancer Controller with annotations for ALB
kubectl annotate service hello-world-frontend service.beta.kubernetes.io/aws-load-balancer-type="nlb" -n dynamo-cloud
```

### Option 2: Ingress with ALB
Use AWS Load Balancer Controller for Application Load Balancer:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: hello-world-ingress
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
spec:
  rules:
  - host: hello-world.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: hello-world-frontend
            port:
              number: 8000
```

**Note**: The hello-world example is primarily for testing internal Dynamo functionality. For production inference workloads, consider the other examples (vLLM, SGLang, TensorRT-LLM).

## Cleanup

```bash
# Remove hello-world deployment
kubectl delete dynamographdeployment hello-world -n dynamo-cloud

# Verify cleanup
kubectl get pods -n dynamo-cloud -l app=hello-world

# Should show no resources found
```

## Next Steps

After successfully running hello-world:

1. **Explore Container Contents**: Use the commands above to examine the full Dynamo codebase
2. **Try GPU Examples**: Move on to `vllm` or `sglang` examples for inference
3. **Custom Development**: Use hello-world as a template for custom services
4. **Study Architecture**: Understand service discovery and communication patterns

## References

- [Dynamo Runtime Hello World Source](https://github.com/ai-dynamo/dynamo/tree/main/examples/runtime/hello_world)
- [NVIDIA Dynamo Architecture Guide](https://docs.nvidia.com/dynamo/)
- [Dynamo Runtime Documentation](https://docs.nvidia.com/dynamo/latest/runtime/)
