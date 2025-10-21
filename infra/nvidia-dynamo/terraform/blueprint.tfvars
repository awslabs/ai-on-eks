name                             = "dynamo-on-eks"
enable_dynamo_stack              = true
enable_aws_efs_csi_driver        = true
enable_aws_efa_k8s_device_plugin = true # Required for NVIDIA Dynamo high-performance networking
enable_ai_ml_observability_stack = true
dynamo_stack_version             = "v0.5.0"
# region                           = "us-west-2"  # Uncomment to override default
# eks_cluster_version              = "1.33"  # Uncomment to override default
