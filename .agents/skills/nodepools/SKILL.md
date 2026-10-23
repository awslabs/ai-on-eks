---
name: nodepools
description: Create a new Karpenter NodePool in this project. Use when adding or configuring EC2 instance provisioning for workloads on EKS, including standard Karpenter and EKS Auto Mode nodepools, EFA networking for P-family instances, and toggle strategies.
---

# Create a NodePool

NodePools define how Karpenter provisions EC2 instances for workloads on EKS.

## Project Layout

All nodepool files live under `infra/base/terraform/`:

```
infra/base/terraform/
├── nodepools.tf                              # Locals (default_nodepools map, merge logic, EFA instance types)
├── variables.tf                              # The `nodepools` map(bool) variable
├── karpenter-resources/
│   ├── karpenter/                            # NodePool + EC2NodeClass manifests (standard Karpenter)
│   │   ├── default.yaml
│   │   ├── gpu.yaml
│   │   ├── neuron.yaml
│   │   └── <your-nodepool>.yaml
│   ├── auto-mode/                            # NodePool + NodeClass manifests (EKS Auto Mode)
│   │   ├── default.yaml
│   │   ├── gpu.yaml
│   │   └── neuron.yaml
│   └── templates/                            # EC2NodeClass spec templates (Karpenter mode only)
│       ├── al2023_ec2nodeclass.tpl
│       └── bottlerocket_ec2nodeclass.tpl
└── modules/
    └── efa-networkinterfaces-generator/      # Generates EFA network interface configs for P-family instances
```

## How the System Works

The nodepool system uses **convention-based file discovery**:

1. Drop a YAML file into `karpenter-resources/karpenter/` (or `auto-mode/`)
2. Terraform discovers all `*.yaml` files in the directory
3. Each file is filtered by name against the `nodepools` map variable
4. Files whose name is NOT in the map default to **always deployed**

The filtering logic in `nodepools.tf`:

```hcl
lookup(local.nodepools, "<filename-without-extension>", true)
#                                                        ^^^^
#                                          default = true (deploy if key not in map)
```

## Two Modes

| Mode | Directory | NodeClass API | Selected by |
|------|-----------|---------------|-------------|
| Standard Karpenter (default) | `karpenter-resources/karpenter/` | `karpenter.k8s.aws/v1` EC2NodeClass | `enable_eks_auto_mode = false` |
| EKS Auto Mode | `karpenter-resources/auto-mode/` | `eks.amazonaws.com/v1` NodeClass | `enable_eks_auto_mode = true` |

## Available Template Variables

All YAML files are processed as Terraform templates. These variables are injected automatically:

| Variable | Description |
|----------|-------------|
| `${node_iam_role}` | IAM role name for nodes |
| `${cluster_name}` | EKS cluster name |
| `${cluster_security_group_id}` | Cluster primary security group ID |
| `${ami_family}` | AMI family (`bottlerocket` or `al2023`) |
| `${ec2nodeclass}` | Pre-rendered EC2NodeClass spec block (Karpenter mode only) |
| `${efa_network_interfaces}` | JSON-encoded map of instance type → EFA networkInterfaces YAML |

---

## Patterns

There are five approaches to creating a nodepool. Choose based on your toggle requirements.

---

### Pattern A: Always-On NodePool (zero code changes)

Use when the nodepool should deploy in every environment unconditionally.

**Steps:**
1. Create a YAML file — that's it.

**File:** `karpenter-resources/karpenter/<name>.yaml`

```yaml
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: <name>
spec:
  disruption:
    budgets:
      - nodes: 10%
    consolidateAfter: 600s
    consolidationPolicy: WhenEmptyOrUnderutilized
  template:
    metadata:
      labels:
        amiFamily: ${ami_family}
        workload-type: <name>
    spec:
      expireAfter: 720h
      nodeClassRef:
        group: karpenter.k8s.aws
        kind: EC2NodeClass
        name: <name>
      requirements:
        - key: karpenter.k8s.aws/instance-category
          operator: In
          values:
            - <category>    # c, m, r, g, p, etc.
        - key: karpenter.k8s.aws/instance-generation
          operator: Gt
          values:
            - "4"
        - key: karpenter.sh/capacity-type
          operator: In
          values:
            - on-demand
      taints:
        - effect: NoSchedule
          key: workload-type
          value: <name>
      terminationGracePeriod: 48h
---
apiVersion: karpenter.k8s.aws/v1
kind: EC2NodeClass
metadata:
  name: <name>
spec:
  ${indent(2, ec2nodeclass)}
```

Since `<name>` is not in `default_nodepools`, the lookup returns `true` and it deploys automatically.

---

### Pattern B: Togglable NodePool (no code changes, controlled via tfvars)

Use when you want per-environment control without modifying Terraform code.

**Steps:**
1. Create the YAML file (same as Pattern A)
2. Add the toggle to your environment's tfvars

**In your `.tfvars`:**

```hcl
nodepools = {
  gpu       = true
  <name>    = true    # explicitly enable
}
```

To disable in a specific environment:

```hcl
nodepools = {
  <name> = false    # skip this nodepool here
}
```

---

### Pattern C: Opt-In NodePool (code change required, off by default)

Use when the nodepool should be **disabled by default** across all environments and only deployed when explicitly enabled.

**Steps:**
1. Create the YAML file (same as Pattern A)
2. Add to `default_nodepools` in `nodepools.tf` with `false`
3. Enable in specific environment tfvars

**In `nodepools.tf`:**

```hcl
locals {
  default_nodepools = {
    gpu       = true
    neuron    = true
    <name>    = false   # off by default, opt-in per environment
  }

  nodepools = merge(local.default_nodepools, var.nodepools)
}
```

**In your `.tfvars`:**

```hcl
nodepools = {
  <name> = true
}
```

---

### Pattern D: P-Family NodePool with EFA (dynamic network interfaces)

Use for P-family GPU instances (p5, p5e, p5en, p6-b200, p6-b300, p6e-gb200) that require EFA network interface configurations.

**Steps:**
1. Add the instance type to `local.efa_instance_types` in `nodepools.tf` (if not already present)
2. Create the YAML file using the `efa_network_interfaces` variable

**Single instance type — file:** `karpenter-resources/karpenter/gpu-<instance-family>.yaml`

```yaml
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: gpu-<instance-family>
spec:
  disruption:
    budgets:
      - nodes: 10%
    consolidateAfter: 600s
    consolidationPolicy: WhenEmptyOrUnderutilized
  template:
    metadata:
      labels:
        amiFamily: ${ami_family}
        accelerator: nvidia
    spec:
      expireAfter: 720h
      nodeClassRef:
        group: karpenter.k8s.aws
        kind: EC2NodeClass
        name: gpu-<instance-family>
      requirements:
        - key: node.kubernetes.io/instance-type
          operator: In
          values:
            - "<instance-type>"       # e.g. "p5.48xlarge"
        - key: karpenter.sh/capacity-type
          operator: In
          values:
            - on-demand
      taints:
        - effect: NoSchedule
          key: nvidia.com/gpu
      terminationGracePeriod: 48h
---
apiVersion: karpenter.k8s.aws/v1
kind: EC2NodeClass
metadata:
  name: gpu-<instance-family>
spec:
  ${indent(2, ec2nodeclass)}
  ${indent(2, lookup(jsondecode(efa_network_interfaces), "<instance-type>", ""))}
```

**Loop over all EFA instance types** (generates one NodePool + EC2NodeClass per type):

```yaml
%{ for instance_type, network_interfaces_yaml in jsondecode(efa_network_interfaces) ~}
---
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: gpu-${replace(instance_type, ".", "-")}
spec:
  disruption:
    budgets:
      - nodes: 10%
    consolidateAfter: 600s
    consolidationPolicy: WhenEmptyOrUnderutilized
  template:
    metadata:
      labels:
        amiFamily: ${ami_family}
        accelerator: nvidia
    spec:
      expireAfter: 720h
      nodeClassRef:
        group: karpenter.k8s.aws
        kind: EC2NodeClass
        name: gpu-${replace(instance_type, ".", "-")}
      requirements:
        - key: node.kubernetes.io/instance-type
          operator: In
          values:
            - "${instance_type}"
        - key: karpenter.sh/capacity-type
          operator: In
          values:
            - on-demand
      taints:
        - effect: NoSchedule
          key: nvidia.com/gpu
      terminationGracePeriod: 48h
---
apiVersion: karpenter.k8s.aws/v1
kind: EC2NodeClass
metadata:
  name: gpu-${replace(instance_type, ".", "-")}
spec:
  ${indent(2, ec2nodeclass)}
  ${indent(2, network_interfaces_yaml)}
%{ endfor ~}
```

**Adding a new EFA instance type to `nodepools.tf`:**

```hcl
locals {
  efa_instance_types = ["p5.48xlarge", "p5e.48xlarge", "p5en.48xlarge", "p6-b200.48xlarge", "p6-b300.48xlarge", "p6e-gb200.36xlarge", "<new-instance-type>"]
}
```

---

### Pattern E: EKS Auto Mode NodePool

Use when the cluster runs with `enable_eks_auto_mode = true`. Auto Mode uses `eks.amazonaws.com/v1` NodeClass with inline specs instead of the shared `${ec2nodeclass}` template.

**File:** `karpenter-resources/auto-mode/<name>.yaml`

```yaml
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: <name>
spec:
  disruption:
    budgets:
      - nodes: 10%
    consolidateAfter: 600s
    consolidationPolicy: WhenEmptyOrUnderutilized
  template:
    metadata:
      labels:
        workload-type: <name>
    spec:
      expireAfter: 336h
      nodeClassRef:
        group: eks.amazonaws.com
        kind: NodeClass
        name: <name>
      requirements:
        - key: eks.amazonaws.com/instance-category
          operator: In
          values:
            - <category>
        - key: eks.amazonaws.com/instance-generation
          operator: Gt
          values:
            - "4"
        - key: karpenter.sh/capacity-type
          operator: In
          values:
            - on-demand
      taints:
        - effect: NoSchedule
          key: workload-type
          value: <name>
      terminationGracePeriod: 48h
---
apiVersion: eks.amazonaws.com/v1
kind: NodeClass
metadata:
  name: <name>
spec:
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
```

---

## Naming Conventions

- YAML file name: kebab-case (`high-memory.yaml`, `gpu-p5.yaml`)
- NodePool `metadata.name`: matches file name (kebab-case)
- EC2NodeClass `metadata.name`: matches the NodePool name
- Toggle key in `default_nodepools` / tfvars: matches file name (`high-memory`, `gpu-p5`)
- Labels: use `workload-type: <name>` or `accelerator: <type>`

## Taint Conventions

| Workload Type | Taint Key | Taint Value | Effect |
|---------------|-----------|-------------|--------|
| NVIDIA GPU | `nvidia.com/gpu` | (none) | NoSchedule |
| AWS Neuron | `aws.amazon.com/neuron` | (none) | NoSchedule |
| Custom workload | `workload-type` | `<name>` | NoSchedule |

## Decision Table

| Scenario | YAML File | tfvars | Code Change |
|----------|-----------|--------|-------------|
| Always-on, all environments | ✅ Drop file | — | None |
| On by default, disable in specific envs | ✅ Drop file | Set `false` where needed | None |
| Explicitly enabled, can toggle per env | ✅ Drop file | Set `true` where wanted | None |
| Off by default, opt-in per env | ✅ Drop file | Set `true` where wanted | Add to `default_nodepools` with `false` |

## Built-in NodePools

| Name | File | Instance Categories | Default | Taint |
|------|------|---------------------|---------|-------|
| `default` | `default.yaml` | c, m, r | Always on (not in map) | None |
| `gpu` | `gpu.yaml` | g (gen > 4) | `true` | `nvidia.com/gpu:NoSchedule` |
| `neuron` | `neuron.yaml` | inf2, trn1, trn2 | `true` | `aws.amazon.com/neuron:NoSchedule` |

## Using NodePools in a Blueprint

Blueprints are self-contained deployments that copy the base Terraform and overlay custom resources. Custom nodepools live in a `nodepools/` directory at the blueprint level and are copied into the Terraform working directory at install time.

### Blueprint Directory Structure

```
infra/<blueprint-name>/
├── install.sh                                # Copies base + overlays custom files
├── nodepools/                                # Custom nodepool YAMLs for this blueprint
│   ├── <custom-nodepool-1>.yaml
│   └── <custom-nodepool-2>.yaml
└── terraform/
    ├── blueprint.tfvars                      # Blueprint-specific variable overrides
    └── _LOCAL/                               # Working directory (populated by install.sh)
        └── karpenter-resources/karpenter/    # Custom YAMLs end up here after install
```

### Step 1: Create the Blueprint `install.sh`

The install script copies the base Terraform into `./terraform/_LOCAL`, then overlays the blueprint's custom nodepool YAMLs:

```bash
#!/bin/bash
# Copy the base into the folder
mkdir -p ./terraform/_LOCAL
cp -r ../base/terraform/* ./terraform/_LOCAL

# Copy custom Karpenter nodepools
cp nodepools/*.yaml ./terraform/_LOCAL/karpenter-resources/karpenter/

cd terraform/_LOCAL
source ./install.sh
```

This ensures the custom nodepools are present alongside the built-in ones (`default.yaml`, `gpu.yaml`, `neuron.yaml`) when Terraform runs.

### Step 2: Create Custom NodePool YAMLs in `nodepools/`

Place your custom nodepool YAML files in the blueprint's `nodepools/` directory. These follow the same format as the base nodepool templates — all Terraform template variables are available.

**Example — Custom NodePool using shared EC2NodeClass** (`nodepools/training.yaml`):

Using `${ec2nodeclass}` is the simplest approach — it inherits the shared node config (AMI, subnets, SGs, SOCI settings):

```yaml
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: training
spec:
  disruption:
    budgets:
      - nodes: 10%
    consolidateAfter: 600s
    consolidationPolicy: WhenEmptyOrUnderutilized
  template:
    metadata:
      labels:
        ai.eks.amazonaws.com/amiFamily: ${ami_family}
        workload-type: training
    spec:
      expireAfter: 720h
      nodeClassRef:
        group: karpenter.k8s.aws
        kind: EC2NodeClass
        name: training
      requirements:
        - key: karpenter.k8s.aws/instance-category
          operator: In
          values:
            - p
        - key: karpenter.k8s.aws/instance-generation
          operator: Gt
          values:
            - "4"
        - key: karpenter.sh/capacity-type
          operator: In
          values:
            - on-demand
      taints:
        - effect: NoSchedule
          key: workload-type
          value: training
      terminationGracePeriod: 48h
---
apiVersion: karpenter.k8s.aws/v1
kind: EC2NodeClass
metadata:
  name: training
spec:
  ${indent(2, ec2nodeclass)}
```

**Example — Custom NodePool with Inline EC2NodeClass** (`nodepools/training-custom.yaml`):

Use an inline EC2NodeClass when you need custom `userData`, `instanceStorePolicy`, `blockDeviceMappings`, or specific `amiSelectorTerms`:

```yaml
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: training-custom
spec:
  disruption:
    budgets:
      - nodes: 10%
    consolidateAfter: 600s
    consolidationPolicy: WhenEmptyOrUnderutilized
  template:
    metadata:
      labels:
        ai.eks.amazonaws.com/amiFamily: ${ami_family}
        workload-type: training
    spec:
      expireAfter: 720h
      nodeClassRef:
        group: karpenter.k8s.aws
        kind: EC2NodeClass
        name: training-custom
      requirements:
        - key: karpenter.k8s.aws/instance-category
          operator: In
          values:
            - p
        - key: karpenter.k8s.aws/instance-generation
          operator: Gt
          values:
            - "4"
        - key: karpenter.sh/capacity-type
          operator: In
          values:
            - on-demand
      taints:
        - effect: NoSchedule
          key: workload-type
          value: training
      terminationGracePeriod: 48h
---
apiVersion: karpenter.k8s.aws/v1
kind: EC2NodeClass
metadata:
  name: training-custom
spec:
  amiFamily: "AL2023"
  amiSelectorTerms:
    - alias: al2023@latest
  role: ${node_iam_role}
  subnetSelectorTerms:
    - tags:
        karpenter.sh/discovery: "${cluster_name}"
        Name: "${cluster_name}-private-secondary*"
  securityGroupSelectorTerms:
    - tags:
        karpenter.sh/discovery: "${cluster_name}"
  instanceStorePolicy: RAID0
  blockDeviceMappings:
    - deviceName: /dev/xvda
      ebs:
        volumeSize: 300Gi
        volumeType: gp3
        encrypted: true
  userData: |
    apiVersion: node.eks.aws/v1alpha1
    kind: NodeConfig
    spec:
      featureGates:
        FastImagePull: true
```

### Step 3: Create `blueprint.tfvars`

Define blueprint-specific variables in `terraform/blueprint.tfvars`:

```hcl
name                             = "<blueprint-name>"
...
...
```

No `nodepools` map override is needed — custom YAML files not in `default_nodepools` are deployed automatically (lookup defaults to `true`). Add a `nodepools` override only if you want to disable built-in nodepools or explicitly toggle your custom ones.

### Step 4: Deploy

```bash
cd infra/<blueprint-name>
./install.sh
```

### Blueprint NodePool Design Guidelines

- **Use `${ec2nodeclass}` when possible** — inherits the shared node config (AMI, subnets, SGs, SOCI settings). Only go inline when you need custom settings.
- **Use inline EC2NodeClass when you need**: custom `userData`, `instanceStorePolicy`, custom `blockDeviceMappings`, or specific `amiSelectorTerms`.
- **EFA is required for P-family**: always include `${indent(2, lookup(jsondecode(efa_network_interfaces), "<instance-type>", ""))}` in the EC2NodeClass spec for P-family instances.
- **Template variables still work**: all files in `karpenter-resources/karpenter/` are processed as Terraform templates regardless of whether they came from base or were copied in by the blueprint.

---

## Creating NodePools Outside Terraform (kubectl + envsubst)

For ad-hoc testing or GitOps-managed nodepools outside Terraform state:

**1. Export cluster values:**

```bash
eval "$(terraform -chdir="./terraform" output -json | jq -r '
  "export CLUSTER_NAME=" + (.deployment_name.value),
  "export KARPENTER_NODE_IAM_ROLE=" + (.karpenter_node_iam_role.value),
  "export SG_ID=" + (.cluster_primary_security_group_id.value)
')"
```

**2. Create YAML with shell-style `${VAR}` placeholders** (not Terraform `${var}`)

**3. Apply:**

```bash
envsubst < my-nodepool.yaml | kubectl apply -f -
```

**Note:** This bypasses Terraform state. The nodepool won't be managed by `terraform destroy`. Use this for iteration/testing, then move the file into `karpenter-resources/` with Terraform template variables for permanent management.

## Checklist

When creating a new nodepool, verify:

- [ ] YAML file placed in correct directory (`karpenter/` or `auto-mode/`)
- [ ] `metadata.name` in NodePool and EC2NodeClass/NodeClass match
- [ ] `nodeClassRef.name` in NodePool spec matches the EC2NodeClass/NodeClass name
- [ ] Using `${indent(2, ec2nodeclass)}` for Karpenter mode EC2NodeClass spec (or inline for Auto Mode)
- [ ] Appropriate taints set (so workloads must tolerate them)
- [ ] Labels applied for workload scheduling (`workload-type`, `accelerator`, `amiFamily`)
- [ ] Decided on toggle strategy (Pattern A/B/C) and applied accordingly
- [ ] For P-family instances: EFA network interfaces included via `${efa_network_interfaces}`
- [ ] `terminationGracePeriod` set appropriately (longer for training workloads)
