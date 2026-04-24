name                             = "dynamo-on-eks"
enable_dynamo_platform           = true
enable_aws_efs_csi_driver        = true
enable_aws_efa_k8s_device_plugin = true
enable_kube_prometheus_stack     = true
enable_soci_snapshotter          = true
enable_nvidia_gpu_operator       = true
availability_zones_count         = 3
# region                           = "us-west-2"
# eks_cluster_version              = "1.34"  # Uncomment to override default

enable_cluster_addons = {
  coredns                         = true
  kube-proxy                      = true
  vpc-cni                         = true
  eks-pod-identity-agent          = true
  metrics-server                  = false
  eks-node-monitoring-agent       = false
  amazon-cloudwatch-observability = false
}

karpenter_additional_ec2nodeclassnames = ["p5-nvidia", "p5e-nvidia", "p5en-nvidia"]
