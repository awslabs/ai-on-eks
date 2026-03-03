name                        = "aioeks-agents"
enable_langfuse             = true        # Enable Langfuse for observability
enable_gitlab               = true        # Enable Gitlab for source control
enable_external_dns         = true        # Enable External DNS to sync Route53 records
enable_milvus               = true        # Enable Milvus for persistent memory storage
enable_mcp_gateway_registry = true        # Enable MCP Gateway Registry
max_user_namespaces         = 16384       # Enables docker builds on bottlerocket
acm_certificate_domain      = ""          # Update with domain (ex agents.example.com)
allowed_inbound_cidrs       = "0.0.0.0/0" # Set the CIDR range allowed through the load balancer. Should include your IP as well as the IP of the gitlab runner node
# region              = "us-west-2"
# eks_cluster_version = "1.34"

# -------------------------------------------------------------------------------------
# EKS Addons Configuration
#
# These are the EKS Cluster Addons managed by Terraform stack.
# You can enable or disable any addon by setting the value to `true` or `false`.
#
# If you need to add a new addon that isn't listed here:
# 1. Add the addon name to the `enable_cluster_addons` variable in `base/terraform/variables.tf`
# 2. Update the `locals.cluster_addons` logic in `eks.tf` to include any required configuration
#
# -------------------------------------------------------------------------------------

# enable_cluster_addons = {
#   coredns                         = true
#   kube-proxy                      = true
#   vpc-cni                         = true
#   eks-pod-identity-agent          = true
#   metrics-server                  = true
#   eks-node-monitoring-agent       = false
#   amazon-cloudwatch-observability = true
# }
