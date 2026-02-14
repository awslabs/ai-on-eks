# EKS Auto Mode Migration Guide

## Overview
This guide documents the changes made to ensure compatibility with EKS Auto Mode, particularly focusing on storage and ingress configurations.

## Key Changes Made

### 1. Storage Class Updates

EKS Auto Mode requires storage classes to use `ebs.csi.eks.amazonaws.com` as the provisioner instead of the standard `ebs.csi.aws.com`.

#### Files Updated:
- `base_eks_setup/gp3.yaml`
- `milvus/ebs-storage-class.yaml`

#### Changes:
```yaml
# Before (Standard EKS)
provisioner: ebs.csi.aws.com

# After (EKS Auto Mode)
provisioner: ebs.csi.eks.amazonaws.com
```

### 2. Ingress Configuration Updates

EKS Auto Mode requires modern ingress configurations without deprecated annotations.

#### Files Updated:
- `model-gateway/litellm-ingress.yaml`
- `model-observability/langfuse-web-ingress.yaml`
- `agentic-apps/strandsdk_agentic_rag_opensearch/k8s/main-app-deployment.yaml`

#### Changes:
```yaml
# Before (Deprecated)
metadata:
  annotations:
    kubernetes.io/ingress.class: alb
spec:
  rules: ...

# After (EKS Auto Mode Compatible)
metadata:
  annotations:
    # kubernetes.io/ingress.class: alb  # REMOVED
spec:
  ingressClassName: alb  # ADDED
  rules: ...
```

### 3. Affected PersistentVolumeClaims

The following PVCs will automatically use the updated storage classes:

#### Files with PVCs using `gp3` storage class:
- `model-hosting/standalone-llamacpp-embedding.yaml` (100Gi)
- `model-hosting/standalone-vllm-reasoning.yaml` (900Gi)
- `model-hosting/standalone-vllm-vision.yaml` (900Gi)

#### Files with volumeClaimTemplates (using default storage class):
- `model-gateway/litellm-deployment.yaml` (postgres StatefulSet - 1Gi)

## Migration Steps for Existing Volumes

### For New Deployments
No additional steps required. The updated storage classes will be used automatically.

### For Existing Deployments with Volumes

If you have existing PersistentVolumes created with the old provisioner (`ebs.csi.aws.com`), you'll need to migrate them:

#### Option 1: Volume Snapshots (Recommended)
1. Create snapshots of existing volumes
2. Delete existing PVCs and PVs
3. Create new PVCs using the updated storage class
4. Restore from snapshots

#### Option 2: Data Migration
1. Create new PVCs with updated storage class
2. Use data migration tools (e.g., `kubectl cp`, `rsync`)
3. Update deployments to use new PVCs
4. Delete old PVCs

### Example Migration Script for Volume Snapshots

```bash
#!/bin/bash

# 1. Create snapshot of existing volume
kubectl patch pvc llamacpp-embedding-server -p '{"metadata":{"finalizers":null}}'

# 2. Create VolumeSnapshot
cat <<EOF | kubectl apply -f -
apiVersion: snapshot.storage.k8s.io/v1
kind: VolumeSnapshot
metadata:
  name: llamacpp-embedding-snapshot
spec:
  source:
    persistentVolumeClaimName: llamacpp-embedding-server
EOF

# 3. Wait for snapshot to be ready
kubectl wait --for=condition=readytouse volumesnapshot/llamacpp-embedding-snapshot --timeout=300s

# 4. Delete old PVC
kubectl delete pvc llamacpp-embedding-server

# 5. Create new PVC from snapshot
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: llamacpp-embedding-server
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: gp3  # This now uses ebs.csi.eks.amazonaws.com
  dataSource:
    name: llamacpp-embedding-snapshot
    kind: VolumeSnapshot
    apiGroup: snapshot.storage.k8s.io
EOF
```

## Verification Steps

### 1. Verify Storage Classes
```bash
kubectl get storageclass
kubectl describe storageclass gp3
kubectl describe storageclass ebs-sc
```

### 2. Verify Ingress Resources
```bash
kubectl get ingress -A
kubectl describe ingress litellm-ingress-alb
kubectl describe ingress langfuse-web-ingress-alb
kubectl describe ingress strandsdk-rag-ingress-alb
```

### 3. Verify PVC Creation
```bash
kubectl get pvc -A
kubectl describe pvc llamacpp-embedding-server
```

## Important Notes

1. **EKS Auto Mode Compatibility**: All changes ensure compatibility with EKS Auto Mode's managed infrastructure.

2. **Backward Compatibility**: The old storage classes and ingress configurations will not work in EKS Auto Mode.

3. **Volume Migration**: Existing volumes created with `ebs.csi.aws.com` must be migrated to use `ebs.csi.eks.amazonaws.com`.

4. **Testing**: Test all deployments in a non-production environment before applying to production.

5. **Monitoring**: Monitor application performance and storage I/O after migration.

## Troubleshooting

### Common Issues

1. **PVC Stuck in Pending**: Check if the storage class exists and uses the correct provisioner.
2. **Ingress Not Creating ALB**: Verify `ingressClassName: alb` is set and deprecated annotations are removed.
3. **Volume Mount Failures**: Ensure the EKS Auto Mode CSI driver is properly configured.

### Debug Commands
```bash
# Check storage class provisioner
kubectl get storageclass -o yaml

# Check PVC events
kubectl describe pvc <pvc-name>

# Check ingress controller logs
kubectl logs -n kube-system -l app.kubernetes.io/name=aws-load-balancer-controller

# Check CSI driver status
kubectl get pods -n kube-system | grep ebs-csi
```

## References

- [EKS Auto Mode Documentation](https://docs.aws.amazon.com/eks/latest/userguide/auto-mode.html)
- [EBS CSI Driver for EKS Auto Mode](https://docs.aws.amazon.com/eks/latest/userguide/ebs-csi.html)
- [AWS Load Balancer Controller](https://kubernetes-sigs.github.io/aws-load-balancer-controller/)