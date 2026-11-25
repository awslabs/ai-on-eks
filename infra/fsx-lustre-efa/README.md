# FSx for Lustre over EFA

Reference architecture that shows how to use Amazon FSx for Lustre with Elastic Fabric Adapter (EFA) from EKS GPU
nodes. It is not tied to a model or an inference engine.

Read [Accelerating model loading with FSx for Lustre over EFA](../../website/docs/guidance/container-startup-time/3-accelerate-model-load-over-efa.md)
first. It lists when FSx for Lustre is and is not the right choice, and the requirements this blueprint satisfies.

## What this deploys

- The base AI on EKS cluster with the FSx CSI driver and the EFA device plugin.
- One Karpenter NodePool and EC2NodeClass per x86 EFA-capable P-family instance type (`nodepools/gpu-p-fsx-efa.yaml`).
  The nodes install the Lustre client and configure it for EFA at boot.

The nodes and the file system both use the EKS cluster security group. Its default rules include all traffic from
itself and all traffic to itself, which is what EFA requires. See the caveats for clusters where the outbound rule is
missing.

It does **not** create the FSx for Lustre file system in Terraform. You create it from Kubernetes with a StorageClass
and PVC, so a file system that cannot be created fails the PVC and not the cluster's Terraform state.

## Deploy

```bash
cd infra/fsx-lustre-efa
./install.sh
```

Edit `terraform/blueprint.tfvars` first to set the region and number of AZs. The region must support EFA-enabled
FSx for Lustre `PERSISTENT_2` file systems.

## Create the file system

FSx for Lustre bills while the file system exists. The script prints the manifests and asks before applying them.

```bash
./scripts/create-fsx-storage.sh -z <availability-zone> -c 4800 -t 1000 -n default
```

Pick the throughput tier that is available in your region. The file system is created in the secondary subnet of
the chosen AZ, which is where the GPU nodes are launched, so the AZ and `/16` CIDR requirements hold. Use the same AZ
for your workload with `topology.kubernetes.io/zone`, `nodeSelector: workload-type: fsx-efa` and a toleration for
`nvidia.com/gpu`.

To apply the manifests yourself, render `examples/storageclass-pvc.yaml` with `envsubst`, or copy it and fill in the
values from `terraform output`.

## Verify EFA is in use

A mounted volume does not prove EFA is used. On a GPU node that has mounted the volume (for example with
`kubectl debug node/<node> --profile=sysadmin --image=public.ecr.aws/amazonlinux/amazonlinux:2023 -it -- chroot /host`):

```bash
sudo lnetctl net show                 # an efa network with the EFA interfaces should be listed
cat /var/log/fsx-efa-client-setup.log # node bootstrap log

# EFA hardware counters (see the EC2 EFA monitoring documentation); note the values, run a read, compare
cat /sys/class/infiniband/<device>/ports/1/hw_counters/rdma_read_bytes

# State of the connections to the storage servers; they should be FULL with few connection attempts
sudo lctl get_param -n osc.*.import | grep -E "state:|connection_attempts"
```

If the counters stay flat while data is read, the data is not using EFA. If a connection stays in `CONNECTING`, reads
stall while the client retries. In our test a file system placed in a different `/16` than the node mounted
successfully, stayed in `CONNECTING` for about six minutes, and then moved data over TCP. Check that the file
system's subnet and the node's subnet share the first two octets of their IP addresses.

The optional smoke test in `examples/read-smoke-test-job.yaml` writes and reads files from a pod. It is not a
benchmark.

## Caveats

- The node bootstrap installs the `lustre-client` package and downloads one AWS script (the EFA configuration) at boot.
  For production, bake both steps into your node image.
- Without the bootstrap, the volume still mounts, but over TCP only. The EKS-optimized AL2023 AMI already has the EFA
  driver and the Lustre and EFA kernel modules; it lacks the Lustre userspace tools and the LNet EFA configuration.
  The node bootstrap here is the shorter sequence: install `lustre-client`, then run the AWS configuration script. We ran
  it on the standard AMI (`c6i.32xlarge`) and on the NVIDIA AMI variant (`p5.48xlarge` Spot, EFA hardware counters moved
  by exactly the 64 GiB read).
- The node pools use `al2023@latest`. Pin the AMI for production.
- A node whose bootstrap failed can still join the cluster without EFA. Verify each node.
- The cluster security group must have an outbound all-traffic rule to itself, in addition to the inbound one. The
  [EKS default cluster security group](https://docs.aws.amazon.com/eks/latest/userguide/sec-group-reqs.html) includes
  both. In our testing, clusters on Kubernetes 1.31 and 1.32 did not have the outbound rule, and with it removed the
  smoke test stalled until it was added back. Check your cluster, and add the rule if it is missing:

  ```bash
  SG=$(terraform -chdir=terraform/_LOCAL output -raw cluster_primary_security_group_id)
  aws ec2 authorize-security-group-egress --group-id "$SG" --ip-permissions "IpProtocol=-1,UserIdGroupPairs=[{GroupId=$SG}]"
  ```
- `p6e-gb200.36xlarge` (arm64) is not included.
- GPU capacity is not guaranteed. The node pools allow reserved, Spot and On-Demand capacity, like the base GPU
  pools, and Karpenter picks the cheapest that is available. If Karpenter reports `InsufficientInstanceCapacity`,
  check which capacity types can be launched: the Spot placement score (`aws ec2 get-spot-placement-scores`)
  describes Spot requests only and says nothing about On-Demand capacity. Otherwise use a capacity reservation (the
  node class selects reservations by the `capacity_reservation_tags` variable) or another type or AZ. Remove `spot`
  from the node pool for workloads that cannot tolerate interruption.
- Creating the file system takes about 7 to 9 minutes. Do not delete the PVC while the file system is still being
  created. The CSI driver can leave the file system behind, still billing, with no PV pointing at it. Wait for the
  PVC to be `Bound`, then delete it.
- Deleting a bound PVC with reclaim policy `Delete` deletes the file system and its data.

## Clean up

Delete the PVCs created with the script first, then run `./cleanup.sh`.
