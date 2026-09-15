amiSelectorTerms:
  - alias: bottlerocket@latest
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
%{ if !enable_soci_snapshotter || soci_snapshotter_use_instance_store ~}
instanceStorePolicy: RAID0
%{ endif ~}
blockDeviceMappings:
  # Root device
  - deviceName: /dev/xvda
    ebs:
      volumeSize: 50Gi
      volumeType: gp3
      encrypted: true
  # Data device: Container resources such as images and logs
  - deviceName: /dev/xvdb
    ebs:
      volumeSize: 300Gi
      volumeType: gp3
%{ if enable_soci_snapshotter && !soci_snapshotter_use_instance_store ~}
      iops: 16000
      throughput: 1000
%{ endif ~}
      encrypted: true
%{ if data_disk_snapshot_id != "" ~}
      snapshotID: ${data_disk_snapshot_id}
%{ endif ~}
userData: |
%{ if enable_soci_snapshotter ~}
  [settings.container-runtime]
  snapshotter = "soci"
  [settings.container-runtime-plugins.soci-snapshotter]
  pull-mode = "parallel-pull-unpack"
  [settings.container-runtime-plugins.soci-snapshotter.parallel-pull-unpack]
  max-concurrent-downloads-per-image = 20
  concurrent-download-chunk-size = "16mb"
  max-concurrent-unpacks-per-image = 10
  discard-unpacked-layers = true
%{ if !soci_snapshotter_use_instance_store ~}
  [settings.bootstrap-commands.k8s-ephemeral-storage]
  commands = [
      ["apiclient", "ephemeral-storage", "init"],
      ["apiclient", "ephemeral-storage" ,"bind", "--dirs", "/var/lib/containerd", "/var/lib/kubelet", "/var/log/pods"]
  ]
  essential = true
  mode = "always"
%{ endif ~}
%{ endif ~}
%{ if max_user_namespaces > 0 ~}
  [settings.kernel.sysctl]
  "user.max_user_namespaces" = "${max_user_namespaces}"
%{ endif ~}
