Summary of Changes Made
Storage Class Updates
Changed: auto-ebs-sc → gp3 in all PersistentVolumeClaim specifications

Reason: EKS Auto Mode uses the gp3 storage class defined in base_eks_setup/gp3.yaml as the default storage class

Node Selector and Toleration Updates
Changed: Node selectors from nvidia.com/gpu: present to nvidia.com/gpu: "true"

Changed: Tolerations from custom keys to standard nvidia.com/gpu key with Exists operator

Reason: EKS Auto Mode uses standard Kubernetes labels and taints for GPU nodes

Files Updated
model-hosting/standalone-vllm-reasoning.yaml

model-hosting/standalone-vllm-vision.yaml

model-hosting/standalone-llamacpp-embedding.yaml

model-gateway/litellm-deployment.yaml

Files Already Compatible
base_eks_setup/gp3.yaml - Already uses correct ebs.csi.eks.amazonaws.com provisioner

milvus/ebs-storage-class.yaml - Already uses correct provisioner

agentic-apps/strandsdk_agentic_rag_opensearch/opensearch-cluster-simple.yaml - CloudFormation template, not affected

These changes ensure all Kubernetes deployments will work correctly with EKS Auto Mode's managed storage and compute infrastructure.