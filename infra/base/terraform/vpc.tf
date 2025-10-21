locals {
  # Calculate CIDR range if not specified. Check if var.vpc_cidr has subnet mask, if not add one based on the following table
  cidr_bits = tomap({
    4 = "19"
    3 = "20"
    2 = "21"
  })
  vpc_cidr = strcontains(var.vpc_cidr, "/") ? var.vpc_cidr : format("%s/%s", var.vpc_cidr, local.cidr_bits[var.availability_zones_count])

  # Calculate subnet sizes based on number of AZs to avoid overlaps
  # We need to allocate space for: private subnets, public subnets, and database subnets
  # Strategy: Divide VPC CIDR into equal blocks for each subnet type

  # Calculate the subnet size needed based on AZ count
  # For 2 AZs: /24 subnets (256 IPs each)
  # For 3 AZs: /25 subnets (128 IPs each)
  # For 4 AZs: /26 subnets (64 IPs each)
  subnet_newbits = var.availability_zones_count == 2 ? 3 : var.availability_zones_count == 3 ? 4 : 5

  # Private subnets: Start from index 0
  # e.g., 10.1.0.0/21 with 2 AZs => ["10.1.0.0/24", "10.1.1.0/24"]
  # e.g., 10.1.0.0/20 with 3 AZs => ["10.1.0.0/25", "10.1.0.128/25", "10.1.1.0/25"]
  private_subnets = [for k, v in local.azs : cidrsubnet(local.vpc_cidr, local.subnet_newbits, k)]

  # Public subnets: Start after private subnets
  # e.g., 10.1.0.0/21 with 2 AZs => ["10.1.2.0/24", "10.1.3.0/24"]
  # e.g., 10.1.0.0/20 with 3 AZs => ["10.1.1.128/25", "10.1.2.0/25", "10.1.2.128/25"]
  public_subnets = [for k, v in local.azs : cidrsubnet(local.vpc_cidr, local.subnet_newbits, k + var.availability_zones_count)]

  # Database subnets: Start after public subnets
  # e.g., 10.1.0.0/21 with 2 AZs => ["10.1.4.0/24", "10.1.5.0/24"]
  # e.g., 10.1.0.0/20 with 3 AZs => ["10.1.3.0/25", "10.1.3.128/25", "10.1.4.0/25"]
  database_private_subnets = var.enable_database_subnets ? [for k, v in local.azs : cidrsubnet(local.vpc_cidr, local.subnet_newbits, k + (2 * var.availability_zones_count))] : []

  # RFC6598 range 100.64.0.0/16 for EKS Data Plane subnets across configurable AZs
  # Divide the secondary CIDR equally among AZs
  # For 2 AZs: /17 subnets (32768 IPs each)
  # For 3 AZs: /18 subnets (16384 IPs each)
  # For 4 AZs: /18 subnets (16384 IPs each) - using only 4 of 4 possible /18 subnets
  secondary_newbits                  = var.availability_zones_count <= 2 ? 1 : 2
  secondary_ip_range_private_subnets = [for k, v in local.azs : cidrsubnet(element(var.secondary_cidr_blocks, 0), local.secondary_newbits, k)]
}

#---------------------------------------------------------------
# VPC
#---------------------------------------------------------------
# WARNING: This VPC module includes the creation of an Internet Gateway and NAT Gateway, which simplifies cluster deployment and testing, primarily intended for sandbox accounts.
# IMPORTANT: For preprod and prod use cases, it is crucial to consult with your security team and AWS architects to design a private infrastructure solution that aligns with your security requirements

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.21"

  name = local.name
  cidr = local.vpc_cidr
  azs  = local.azs

  # Secondary CIDR block attached to VPC for EKS Control Plane ENI + Nodes + Pods
  secondary_cidr_blocks = var.secondary_cidr_blocks

  # 1/ EKS Data Plane secondary CIDR blocks for subnets across configurable AZs for EKS Control Plane ENI + Nodes + Pods
  # 2/ Private Subnets with RFC1918 private IPv4 address range for Private NAT + NLB + Airflow + EC2 Jumphost etc.
  private_subnets = concat(local.private_subnets, local.secondary_ip_range_private_subnets)

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
  version = "~> 5.21"

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
