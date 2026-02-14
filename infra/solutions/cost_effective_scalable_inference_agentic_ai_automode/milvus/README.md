# Milvus Vector Database on AWS EKS with Graviton

This directory contains configuration files for deploying Milvus, a vector database, on Amazon EKS with AWS Graviton processors. This setup is part of the larger project for cost-effective and scalable Small Language Models inference on AWS Graviton4 with EKS.

## Overview

Milvus is an open-source vector database built to power embedding similarity search and AI applications. In this setup, we deploy Milvus in standalone mode on AWS Graviton-based EKS nodes to leverage the cost-effectiveness and performance of ARM64 architecture.

## Prerequisites

- An existing EKS cluster with Graviton (ARM64) nodes
- Cert-manager installed on the cluster
- Milvus Operator installed on the cluster
- AWS EBS CSI driver configured for persistent storage

## Configuration Files

This directory includes the following configuration files:

1. **ebs-storage-class.yaml**: Defines an AWS EBS storage class for Milvus persistent storage
   - Uses gp3 volume type
   - Enables encryption
   - Configures WaitForFirstConsumer binding mode

2. **milvus-standalone.yaml**: Deploys Milvus in standalone mode
   - Configures Milvus to run on ARM64 (Graviton) nodes
   - Sets up resource requests
   - Configures in-cluster dependencies (etcd, pulsar, storage)
   - All components are configured to run on ARM64 architecture

3. **milvus-nlb-service.yaml**: Creates a Network Load Balancer service for external access
   - Exposes Milvus service port (19530)
   - Exposes metrics port (9091)
   - Configures internet-facing NLB

## Deployment Steps

1. **Install cert-manager** (if not already installed):
   ```bash
   kubectl apply -f https://github.com/jetstack/cert-manager/releases/download/v1.5.3/cert-manager.yaml
   kubectl get pods -n cert-manager
   ```

2. **Install Milvus Operator** (if not already installed):
   ```bash
   kubectl apply -f https://raw.githubusercontent.com/zilliztech/milvus-operator/main/deploy/manifests/deployment.yaml
   kubectl get pods -n milvus-operator
   ```

3. **Create EBS Storage Class**:
   ```bash
   kubectl apply -f ebs-storage-class.yaml
   ```

4. **Deploy Milvus in standalone mode**:
   ```bash
   kubectl apply -f milvus-standalone.yaml
   ```

5. **Create NLB Service** (optional, if you need external access):
   ```bash
   kubectl apply -f milvus-nlb-service.yaml
   ```

## Accessing Milvus

You can access Milvus using port-forwarding:
```bash
kubectl port-forward service/my-release-milvus 19530:19530
```

Or through the Network Load Balancer if you deployed the NLB service.

## Integration with LLM Services

This Milvus deployment can be integrated with the LLM services in the parent project for vector search capabilities, enabling:
- Semantic search
- Retrieval-augmented generation (RAG)
- Document similarity matching
- And other vector-based operations

## Uninstalling

To uninstall Milvus:
```bash
kubectl delete milvus my-release
```

## Additional Resources

- [Milvus Documentation](https://milvus.io/docs)
- [Milvus Operator GitHub](https://github.com/zilliztech/milvus-operator)
- [Main Project README](../README.md)
