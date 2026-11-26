name                             = "fsx-lustre-efa"
ami_family                       = "al2023"
enable_aws_fsx_csi_driver        = true
enable_aws_efa_k8s_device_plugin = true
# FSx for Lustre is created from Kubernetes (StorageClass + PVC), not from Terraform.
# deploy_fsx_volume stays at its default (false).

# The EFA-enabled FSx for Lustre file system must be in a region where FSx supports it:
# https://docs.aws.amazon.com/fsx/latest/LustreGuide/using-fsx-lustre.html#persistent-deployment-regions
# region                           = "us-west-2"
# availability_zones_count         = 3
