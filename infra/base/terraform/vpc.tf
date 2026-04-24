data "aws_availability_zones" "available" {
  # only use zones that are availability-zones (e.g: exclude local and wavelength zones)
  filter {
    name   = "zone-type"
    values = ["availability-zone"]
  }
}

data "aws_availability_zones" "available_lz" {
  # local zones
  filter {
    name   = "opt-in-status"
    values = ["opted-in"]
  }
  filter {
    name   = "zone-type"
    values = ["local-zone"]
  }
}

locals {
  region    = var.region
  local_azs = slice(data.aws_availability_zones.available_lz.names, 0, var.local_zones_count)
  az_azs    = slice(data.aws_availability_zones.available.names, 0, var.availability_zones_count)

  # concat to put local zones at the end of the list to avoid creation on NAT Gateway
  azs       = concat(local.az_azs, local.local_azs)
  azs_count = var.availability_zones_count + var.local_zones_count

  vpc_cidr        = var.vpc_cidr
  vpc_cidr_prefix = tonumber(split("/", local.vpc_cidr)[1])

  private_subnets          = [for k, v in module.subnets.network_cidr_blocks : v if endswith(k, "/private")]
  public_subnets           = [for k, v in module.subnets.network_cidr_blocks : v if endswith(k, "/public")]
  control_plane_subnets    = [for k, v in module.subnets.network_cidr_blocks : v if endswith(k, "/cp")]
  database_private_subnets = var.enable_database_subnets ? [for k, v in module.subnets.network_cidr_blocks : v if endswith(k, "/database")] : []

  # Subnet sizing: newbits determines how many subnets fit per secondary CIDR block (/16 CIDR prefix)
  # 1 - 2 /17 subnets (32763 IPs each)
  # 2 - 4 /18 subnets (16379 IPs each)
  # 3 - 8 /19 subnets (8187 IPs each)
  secondary_newbits = 1

  # Max subnets that fit in one CIDR block given the newbits
  secondary_subnets_per_cidr = pow(2, local.secondary_newbits)

  # For each AZ, pick the CIDR block (first one until full, then overflow to next) and compute the subnet index within that block
  secondary_ip_range_private_subnets = [
    for k, v in local.azs : cidrsubnet(
      var.secondary_cidr_blocks[floor(k / local.secondary_subnets_per_cidr)],
      local.secondary_newbits,
      k % local.secondary_subnets_per_cidr
    )
  ]
}

module "subnets" {
  source  = "hashicorp/subnets/cidr"
  version = "1.0.0"

  base_cidr_block = var.vpc_cidr
  networks = concat(
    [for k, v in local.azs : tomap({ "name" = "${v}/public", "new_bits" = var.public_subnets_cidr_prefix - local.vpc_cidr_prefix })],
    [for k, v in local.azs : tomap({ "name" = "${v}/private", "new_bits" = var.private_subnets_cidr_prefix - local.vpc_cidr_prefix })],
    [for k, v in local.azs : tomap({ "name" = "${v}/cp", "new_bits" = var.control_plane_subnets_cidr_prefix - local.vpc_cidr_prefix })],
    var.enable_database_subnets ? [for k, v in local.azs : tomap({ "name" = "${v}/database", "new_bits" = var.database_subnets_cidr_prefix - local.vpc_cidr_prefix })] : []
  )
}


#---------------------------------------------------------------
# VPC
#---------------------------------------------------------------
# WARNING: This VPC module includes the creation of an Internet Gateway and NAT Gateway, which simplifies cluster deployment and testing, primarily intended for sandbox accounts.
# IMPORTANT: For preprod and prod use cases, it is crucial to consult with your security team and AWS architects to design a private infrastructure solution that aligns with your security requirements

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 6.4"

  name = local.name
  cidr = local.vpc_cidr
  azs  = local.azs

  # Secondary CIDR block attached to VPC for EKS Control Plane ENI + Nodes + Pods
  secondary_cidr_blocks = var.secondary_cidr_blocks

  # 1/ EKS Data Plane secondary CIDR blocks for subnets across configurable AZs for EKS Control Plane ENI + Nodes + Pods
  # 2/ Private Subnets with RFC1918 private IPv4 address range for Private NAT + NLB + Airflow + EC2 Jumphost etc.
  private_subnets = concat(local.private_subnets, local.secondary_ip_range_private_subnets)

  # EKS Control Plane subnets
  intra_subnets = local.control_plane_subnets

  # ------------------------------
  # Private Subnets for MLflow backend store
  database_subnets                   = local.database_private_subnets
  create_database_subnet_group       = var.enable_database_subnets
  create_database_subnet_route_table = var.enable_database_subnets

  # ------------------------------
  # Optional Public Subnets for NAT and IGW for PoC/Dev/Test environments
  # Public Subnets can be disabled while deploying to Production and use Private NAT + TGW
  public_subnets     = local.public_subnets
  enable_nat_gateway = true
  single_nat_gateway = var.single_nat_gateway
  #-------------------------------

  public_subnet_tags = {
    "kubernetes.io/role/elb" = 1
  }
  private_subnet_names = concat(
    [for k, v in local.azs : "${var.name}-private-${v}"],
    [for k, v in local.azs : "${var.name}-private-secondary-${v}"]
  )
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = 1
    # Tags subnets for Karpenter auto-discovery
    "karpenter.sh/discovery" = local.name
  }

  tags = local.tags
}

################################################################################
# VPC Endpoints
################################################################################

module "vpc_endpoints" {
  source  = "terraform-aws-modules/vpc/aws//modules/vpc-endpoints"
  version = "~> 6.4"

  vpc_id = module.vpc.vpc_id

  endpoints = {
    s3 = {
      service         = "s3"
      service_type    = "Gateway"
      route_table_ids = module.vpc.private_route_table_ids
      tags = {
        Name = "${var.name}-s3-vpc-endpoint"
      }
    }
    s3express = {
      service         = "s3express"
      service_type    = "Gateway"
      route_table_ids = module.vpc.private_route_table_ids
      tags = {
        Name = "${var.name}-s3express-vpc-endpoint"
      }
    }
  }

  tags = local.tags
}
