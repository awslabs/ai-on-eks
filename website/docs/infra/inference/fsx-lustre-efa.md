---
sidebar_position: 3
sidebar_label: FSx for Lustre over EFA
---

# FSx for Lustre over EFA on EKS

This reference architecture shows how to use Amazon FSx for Lustre with Elastic Fabric Adapter (EFA) from EKS GPU
nodes. It is not tied to a model or an inference engine.

Read [Accelerating model loading with FSx for Lustre over EFA](../../guidance/container-startup-time/3-accelerate-model-load-over-efa.md)
first. It explains when FSx for Lustre is and is not the right choice, and lists the requirements this architecture
satisfies.

## What it deploys

- The base AI on EKS cluster with the FSx CSI driver and the EFA device plugin enabled.
- One Karpenter NodePool and EC2NodeClass per x86 EFA-capable P-family instance type. The nodes install the Lustre
  client and configure it for EFA at boot.

The nodes and the file system both use the EKS cluster security group, whose default rules include all traffic from
itself and all traffic to itself, as EFA requires.

The FSx for Lustre file system is **not** created in Terraform. You create it from Kubernetes with a StorageClass and
a PersistentVolumeClaim, so a file system that cannot be created fails the PVC and not the cluster's Terraform state.

## Deploy

```bash
cd infra/fsx-lustre-efa
./install.sh
```

Set the region and the number of availability zones in `terraform/blueprint.tfvars` first. The region must support
EFA-enabled FSx for Lustre `PERSISTENT_2` file systems.

## Create the file system

FSx for Lustre bills while the file system exists. The helper script prints the manifests and asks before applying
them.

```bash
./scripts/create-fsx-storage.sh -z <availability-zone> -c 4800 -t 1000
```

The file system is created in the secondary subnet of the chosen availability zone, which is where the GPU nodes are
launched, so the same-AZ and same-`/16` requirements hold. Schedule your workload in the same zone with
`topology.kubernetes.io/zone`, the node selector `workload-type: fsx-efa` and a toleration for `nvidia.com/gpu`.

## Verify EFA is in use

A mounted volume does not prove that EFA is used. On a GPU node that has mounted the volume:

```bash
sudo lnetctl net show      # an efa network with the EFA interfaces should be listed
rdma -p statistic show     # rdma_read_bytes should increase while data is read
```

## Caveats

- The node bootstrap installs the `lustre-client` package and downloads one AWS script (the EFA configuration) at boot.
  For production, bake both steps into your node image.
- Without the bootstrap, the volume still mounts, but over TCP only. The EKS-optimized AL2023 AMI already has the EFA
  driver and the Lustre and EFA kernel modules; it lacks the Lustre userspace tools and the LNet EFA configuration. See
  the guide for what we found on the standard and the NVIDIA AMI variants.
- The node pools use `al2023@latest`. Pin the AMI for production.
- A node whose bootstrap failed still joins the cluster without EFA. Verify every node.
- The cluster security group must have an outbound all-traffic rule to itself, in addition to the inbound one. The
  [EKS default cluster security group](https://docs.aws.amazon.com/eks/latest/userguide/sec-group-reqs.html) includes
  both, but in our testing clusters on Kubernetes 1.31 and 1.32 did not have the outbound rule. The README shows how to
  check for it and add it.
- Deleting a PVC with reclaim policy `Delete` deletes the file system and its data.

The full source and the README are in the
[`infra/fsx-lustre-efa`](https://github.com/awslabs/ai-on-eks/tree/main/infra/fsx-lustre-efa) directory.
