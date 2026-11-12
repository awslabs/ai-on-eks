# Base configuration for the multi-node Llama-3.1-405B FSx-over-EFA blueprint.
# Serves BOTH blueprint variants (../../blueprints/inference/vllm-llama3.1-405b-lws-gpu-fsx and -dlc)
# — they share the same infra; only the image/manifest differs.
# Deploy: run ../install.sh (copies the base into terraform/_LOCAL and applies it with this tfvars).

# Must stay <= 17 chars: the Karpenter controller IAM role name_prefix is
# "KarpenterController-${name}-" and the AWS provider caps name_prefix at 38.
name = "vllm-405b-fsx"

# --- FSx for Lustre over EFA (the purpose of this blueprint) ---
deploy_fsx_volume              = true          # create the FSx for Lustre filesystem + static PV/PVC (fsx-static-pvc)
fsx_efa_enabled                = true          # EFA (RDMA) mode on FSx + the FSx-co-located EFA Karpenter NodePool (gpu-p5-fsx)
enable_aws_fsx_csi_driver      = true          # FSx CSI driver — lets the pods mount fsx-static-pvc
fsx_efa_nodepool_instance_type = "p5.48xlarge" # EFA GPU NodePool instance type (must be one of the base efa_instance_types)
# fsx_storage_capacity         = 19200         # optional: 4 OSTs for more weight-load parallelism (null => 4800, the EFA minimum)

# --- EFA device plugin: advertises vpc.amazonaws.com/efa, which the workload pods request (32 per pod) ---
enable_aws_efa_k8s_device_plugin = true

# NVIDIA device plugin is ON by default in the base (enable_nvidia_device_plugin = true), so
# nvidia.com/gpu is advertised without setting anything here. Do NOT enable the GPU Operator
# (enable_nvidia_gpu_operator) unless intended — it switches off the standalone device plugin.

# region              = "us-west-2"
# eks_cluster_version = "1.34"
