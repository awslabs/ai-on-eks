amiSelectorTerms:
  - alias: al2023@latest
karpenterRole: ${node_iam_role}
subnetSelectorTerms:
  tags:
    karpenter.sh/discovery: "${cluster_name}"
    Name: "${cluster_name}-private-secondary*" # Only seconddary cidr subnets
securityGroupSelectorTerms:
  tags:
    Name: ${cluster_name}-node
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
userData: |
  MIME-Version: 1.0
  Content-Type: multipart/mixed; boundary="//"

  --//
%{ if enable_soci_snapshotter && soci_snapshotter_use_instance_store ~}
  Content-Type: text/x-shellscript; charset="us-ascii"

  #!/bin/bash
  sed -i "s|ExecStart=/usr/bin/soci-snapshotter-grpc|ExecStart=/usr/bin/soci-snapshotter-grpc --root /var/lib/containerd/io.containerd.snapshotter.v1.soci|" /etc/systemd/system/soci-snapshotter.service
  systemctl daemon-reload

  --//
%{ endif ~}
%{ if enable_soci_snapshotter ~}
  Content-Type: application/node.eks.aws

  apiVersion: node.eks.aws/v1alpha1
  kind: NodeConfig
  spec:
    featureGates:
      FastImagePull: true
%{ if soci_snapshotter_use_instance_store ~}
    containerd:
      config: |
        [proxy_plugins.soci.exports]
        root = "/var/lib/containerd/io.containerd.snapshotter.v1.soci"
%{ endif ~}
  --//
%{ endif ~}
