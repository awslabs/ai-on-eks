amiSelectorTerms:
  - alias: al2023@latest
role: ${node_iam_role}
subnetSelectorTerms:
  - tags:
      karpenter.sh/discovery: "${cluster_name}"
      Name: "${cluster_name}-private-secondary*" # Only seconddary cidr subnets
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
