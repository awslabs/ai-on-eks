ephemeralStorage:
  iops: 3000
  size: 80Gi
  throughput: 125
networkPolicy: DefaultAllow
networkPolicyEventLogs: Disabled
role: ${node_iam_role}
securityGroupSelectorTerms:
  - id: ${cluster_security_group_id}
snatPolicy: Random
subnetSelectorTerms:
  - tags:
      karpenter.sh/discovery: "${cluster_name}"
      Name: "${cluster_name}-private-secondary*" # Only secondary cidr subnets
capacityReservationSelectorTerms:
  - tags:
%{ for tag_key, tag_value in jsondecode(capacity_reservation_tags) ~}
      ${tag_key}: "${tag_value}"
%{ endfor ~}
