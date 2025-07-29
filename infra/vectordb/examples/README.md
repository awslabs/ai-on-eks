# VectorDB Examples

This guide provides instructions for deploying and testing various vector databases on EKS using AWS Graviton Instances. These can be used as standalone applications or integrated with other ai-on-eks blueprints.

## Common Prerequisites

Before installing any vector database, ensure you have the following:

- AWS CLI configured with appropriate permissions
- kubectl installed and configured
- Terraform >= 1.0
- An existing EKS cluster or ability to create one
- Helm 3.x installed

## Common Installation Steps

### Step 1: Clone and Navigate to the Repository

```
git clone https://github.com/awslabs/ai-on-eks.git
cd ai-on-eks/infra/vectordb
```

### Step 2: Configure Terraform Variables

Edit the `terraform/blueprint.tfvars` file to enable your desired vector database:

```
# Enable your desired vector database (choose one or more)
enable_vectordb_milvus = true    # For Milvus
enable_vectordb_qdrant = true    # For Qdrant
enable_vectordb_weaviate = true  # For Weaviate
```

### Step 3: Deploy the Infrastructure

Run the installation script to deploy the EKS cluster with vector database support:

```
chmod +x install.sh
./install.sh
```

This script will:
- Copy base Terraform configurations
- Initialize Terraform
- Deploy the EKS cluster with required addons
- Install the selected vector database(s) via ArgoCD addons

### Step 4: Verify EKS Cluster

Confirm your EKS cluster is running:

```
kubectl get nodes
kubectl get namespaces
```

## Common Troubleshooting

### Common Issues and Solutions

1. **Pods stuck in Pending state**
   ```
   kubectl describe pods -n <namespace>
   # Check for resource constraints or node selector issues
   ```

2. **LoadBalancer service not getting external IP**
   ```
   # Check AWS Load Balancer Controller
   kubectl get pods -n kube-system | grep aws-load-balancer
   ```

3. **Connection timeout**
   ```
   # Check security groups and network connectivity
   kubectl get svc <service-name> -n <namespace> -o yaml
   ```

4. **Insufficient resources**
   ```
   # Check node resources
   kubectl top nodes
   kubectl describe nodes
   ```

---

## Milvus

Milvus is a cloud-native vector database designed for storing, indexing, and managing embedding vectors for similarity search and AI applications.

### Milvus-Specific Installation Steps

After completing the common installation steps with `enable_vectordb_milvus = true`:

1. **Verify Milvus Operator Installation**:
   ```
   # Check if milvus-operator namespace exists
   kubectl get namespace milvus
   
   # Verify Milvus operator pods are running
   kubectl get pods -n milvus
   
   # Check ArgoCD applications (if ArgoCD is accessible)
   kubectl get applications -n argocd | grep milvus
   ```

2. **Deploy Milvus Cluster**:
   ```
   # Deploy the Milvus cluster
   kubectl apply -f examples/milvus_distributed.yaml -n milvus
   ```

   Note: The `milvus_distributed.yaml` file contains the Milvus cluster configuration optimized for AWS Graviton instances with ARM64 CPU and SSD storage.

3. **Monitor Deployment**:
   ```
   # Check Milvus custom resource
   kubectl get milvus my-release -n milvus
   
   # Check all pods in milvus namespace
   kubectl get pods -n milvus
   
   ```

4. **Get Connection Details**:
   ```
   kubectl get svc my-release-milvus -n milvus
   ```
### Testing Milvus

Test Milvus connectivity using CURL:

# Set up port forwarding in one terminal
```
kubectl port-forward svc/my-release-milvus 19530:19530 -n milvus
```

# In another terminal, test the connection
```
export CLUSTER_ENDPOINT="http://localhost:19530" # Replace with your Milvus cluster endpoint
export TOKEN="root:Milvus" # Replace with your Milvus username:password
```
# List collections (should return empty list initially)
```
curl --request POST \
  --url "${CLUSTER_ENDPOINT}/v2/vectordb/collections/list" \
  --header 'accept: application/json' \
  --header 'content-type: application/json' \
  -d '{
  "dbName": "default"
  }'
```

If using LoadBalancer in production environment (replace EXTERNAL-IP with actual IP):

### Milvus-Specific Commands

```
# Check Milvus logs
kubectl logs -l app.kubernetes.io/name=milvus -n milvus

# Scale Milvus components
kubectl patch milvus my-release -n milvus --type='merge' -p='{"spec":{"components":{"queryNode":{"replicas":2}}}}'
```

---

## Qdrant

Qdrant is a vector similarity search engine with extended filtering support, designed to handle large collections of vectors.

### Qdrant-Specific Installation Steps

After completing the common installation steps with `enable_vectordb_qdrant = true`:

1. **Verify Qdrant Deployment**:
   
   # Check if qdrant namespace exists
   ```
   kubectl get namespace qdrant
   ```
   # Verify Qdrant pods are running (should see 3 replicas)
   ```
   kubectl get pods -n qdrant
   ```
   
   # Check ArgoCD applications (if ArgoCD is accessible)
   ```
   kubectl get applications -n argocd | grep qdrant
   ```
   
   # Wait for all pods to be ready
   ```
   kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=qdrant -n qdrant --timeout=600s
   ```

2. **Monitor Deployment**:
   
   # Check StatefulSet
   ```
   kubectl get statefulset -n qdrant
   ```
   
   # Check services
   ```
   kubectl get svc -n qdrant
   ```

3. **Get Connection Details**:
   ```
   kubectl get svc qdrant -n qdrant
   ```
   Note the service details - Qdrant will be accessible on port 6333 for HTTP API and port 6334 for gRPC.

### Testing Qdrant

Test Qdrant connectivity using CURL:

# Set up port forwarding in one terminal
```
kubectl port-forward svc/qdrant 6333:6333 -n qdrant
```

# In another terminal, test the connection
# Check cluster info
```
curl -X GET "http://localhost:6333/cluster"
```
# List collections (should return empty result initially)
```
curl -X GET "http://localhost:6333/collections"
```
# Check telemetry/health
```
curl -X GET "http://localhost:6333/telemetry"
```
If using LoadBalancer (replace EXTERNAL-IP with actual IP):


### Qdrant-Specific Commands


# Check Qdrant logs
```
kubectl logs -l app.kubernetes.io/name=qdrant -n qdrant
```

# Port forward for local testing
```
kubectl port-forward svc/qdrant 6333:6333 -n qdrant
```
# Check cluster status via API
```
curl http://localhost:6333/cluster
```
# Scale Qdrant replicas (if needed)
```
kubectl scale statefulset qdrant --replicas=10 -n qdrant
```

---

## Weaviate

Weaviate is an open-source vector search engine that stores both objects and vectors, allowing for combining vector search with structured filtering.

### Weaviate-Specific Installation Steps

After completing the common installation steps with `enable_vectordb_weaviate = true`:

1. **Verify Weaviate Deployment**:
   ```
   # Check if weaviate namespace exists
   kubectl get namespace weaviate
   
   # Verify Weaviate pods are running (should see 4 replicas)
   kubectl get pods -n weaviate
   
   # Check ArgoCD applications (if ArgoCD is accessible)
   kubectl get applications -n argocd | grep weaviate
   
   # Wait for all pods to be ready
   kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=weaviate -n weaviate --timeout=600s
   ```

2. **Monitor Deployment**:
   ```
   # Check StatefulSet
   kubectl get statefulset -n weaviate
   
   # Check services
   kubectl get svc -n weaviate
   ```

3. **Get Connection Details**:
   ```
   kubectl get svc weaviate -n weaviate
   ```
   Note the service details - Weaviate will be accessible on port 8080 for HTTP API.

### Testing Weaviate

Test Weaviate connectivity using CURL:

# Set up port forwarding in one terminal

```
kubectl port-forward svc/weaviate 50052:50052 -n weaviate
```

# In another terminal, test the connection

# Check meta information (version, hostname, etc.)
```
curl -X GET "http://localhost:50052/v1/meta"
```
# List schema classes (should return empty classes initially)
```
curl -X GET "http://localhost:50052/v1/schema"
```
If using LoadBalancer (replace EXTERNAL-IP with actual IP):

### Weaviate-Specific Commands

# Check Weaviate logs

```
kubectl logs -l app.kubernetes.io/name=weaviate -n weaviate
```

# Scale Weaviate replicas (if needed)
```
kubectl scale statefulset weaviate --replicas=6 -n weaviate
```

## Next Steps

- Explore the documentation for your chosen vector database:
  - [Milvus documentation](https://milvus.io/docs)
  - [Qdrant documentation](https://qdrant.tech/documentation/)
  - [Weaviate documentation](https://weaviate.io/developers/weaviate)
- Integrate with your AI/ML applications
- Set up monitoring and alerting
- Configure authentication for production use
- Optimize cluster configuration based on your workload requirements