amiSelectorTerms:
  - alias: al2023@latest
role: ${node_iam_role}
subnetSelectorTerms:
%{ if fsx_subnet_id != "" ~}
  # FSx-over-EFA: pin to the FSx filesystem's subnet. EFA requires the client in FSx's
  # same AZ + /16 CIDR; a secondary-CIDR subnet is a different /16 and breaks the EFA path.
  - id: "${fsx_subnet_id}"
%{ else ~}
  - tags:
      karpenter.sh/discovery: "${cluster_name}"
      Name: "${cluster_name}-private-secondary*" # Only seconddary cidr subnets
%{ endif ~}
securityGroupSelectorTerms:
  - tags:
      karpenter.sh/discovery: "${cluster_name}"
capacityReservationSelectorTerms:
  - tags:
%{ for tag_key, tag_value in jsondecode(capacity_reservation_tags) ~}
      ${tag_key}: "${tag_value}"
%{ endfor ~}
instanceStorePolicy: RAID0
blockDeviceMappings:
  - deviceName: /dev/xvda
    ebs:
      volumeSize: 300Gi
      volumeType: gp3
      encrypted: true
%{ if enable_soci_snapshotter && !soci_snapshotter_use_instance_store ~}
      iops: 16000
      throughput: 1000
%{ endif ~}
%{ if fsx_efa_enabled ~}
userData: |
  MIME-Version: 1.0
  Content-Type: multipart/mixed; boundary="//"
  --//
  Content-Type: text/x-shellscript; charset="us-ascii"
  #!/bin/bash
  # Put the FSx for Lustre client on EFA (LNet/RDMA) instead of TCP; no-op on non-EFA nodes.
  if [ -d /sys/class/infiniband ] && [ -n "$(ls -A /sys/class/infiniband 2>/dev/null)" ]; then
    cd /tmp || exit 0
    curl -fsSL -O https://docs.aws.amazon.com/fsx/latest/LustreGuide/samples/install-fsx-lustre-client.zip \
      && unzip -o install-fsx-lustre-client.zip \
      && ( cd install-fsx-lustre-client && ./bin/install-fsx-lustre-client.sh --install-lustre --install-efa )
    curl -fsSL -O https://docs.aws.amazon.com/fsx/latest/LustreGuide/samples/configure-efa-fsx-lustre-client.zip \
      && unzip -o configure-efa-fsx-lustre-client.zip \
      && ( cd configure-efa-fsx-lustre-client && ./setup.sh )
  fi
%{ if enable_soci_snapshotter ~}
  --//
  Content-Type: application/node.eks.aws
  apiVersion: node.eks.aws/v1alpha1
  kind: NodeConfig
  spec:
    featureGates:
      FastImagePull: true
%{ if !soci_snapshotter_use_instance_store ~}
    instance:
      localStorage:
        disabledMounts:
          - SOCI
%{ endif ~}
%{ endif ~}
  --//--
%{ else ~}
%{ if enable_soci_snapshotter ~}
userData: |
  apiVersion: node.eks.aws/v1alpha1
  kind: NodeConfig
  spec:
    featureGates:
      FastImagePull: true
%{ if !soci_snapshotter_use_instance_store ~}
    instance:
      localStorage:
        disabledMounts:
          - SOCI
%{ endif ~}
%{ endif ~}
%{ endif ~}
