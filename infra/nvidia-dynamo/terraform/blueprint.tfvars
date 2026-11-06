name                             = "dynamo-on-eks"
ami_family                       = "al2023"
enable_dynamo_platform           = true
enable_kai_scheduler             = true
enable_grove                     = true
enable_aws_efs_csi_driver        = true
enable_aws_efa_k8s_device_plugin = true
efa_network_interfaces_policy    = "2"
enable_kube_prometheus_stack     = true
enable_nvidia_gpu_operator       = true
enable_s3_models_storage         = true
s3_models_inference_sa_namespace = "dynamo-system"
s3_models_sync_sa_namespace      = "dynamo-system"
# region                           = "us-west-2"
# eks_cluster_version              = "1.36"  # Uncomment to override default

vpc_cni_configuration = {
  env = {
    ENABLE_PREFIX_DELEGATION     = "true"
    MINIMUM_IP_TARGET            = "16"
    WARM_IP_TARGET               = "5"
    ENABLE_MULTI_NIC             = "true"
    AWS_VPC_K8S_CNI_EXTERNALSNAT = "true"
  }
}
