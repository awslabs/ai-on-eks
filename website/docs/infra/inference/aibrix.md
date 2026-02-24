---
sidebar_label: AIBrix on EKS
---
import CollapsibleContent from '@site/src/components/CollapsibleContent';

# AIBrix on EKS

:::warning
Deployment of ML models on EKS requires access to GPUs or Neuron instances. If your deployment isn't working, it’s often due to missing access to these resources. Also, some deployment patterns rely on Karpenter autoscaling and static node groups; if nodes aren't initializing, check the logs for Karpenter or Node groups to resolve the issue.
:::

:::info
These instructions only deploy the AIBrix cluster as a base. If you are looking to deploy specific models for inference or training, please refer to this [AI](https://awslabs.github.io/ai-on-eks/docs/blueprints) page for end-to-end instructions.
:::

### What is AIBrix?
AIBrix is an open source initiative designed to provide essential building blocks to construct scalable GenAI inference infrastructure. AIBrix delivers a cloud-native solution optimized for deploying, managing, and scaling large language model (LLM) inference, tailored specifically to enterprise needs.
![Alt text](https://aibrix.readthedocs.io/latest/_images/aibrix-architecture-v1.jpeg)

### Key Features and Benefits
* LLM Gateway and Routing: Efficiently manage and direct traffic across multiple models and replicas.
* High-Density LoRA Management: Streamlined support for lightweight, low-rank adaptations of models.
* Distributed Inference: Scalable architecture to handle large workloads across multiple nodes.
* LLM App-Tailored Autoscaler: Dynamically scale inference resources based on real-time demand.
* Unified AI Runtime: A versatile sidecar enabling metric standardization, model downloading, and management.
* Heterogeneous-GPU Inference: Cost-effective SLO-driven LLM inference using heterogeneous GPUs.
* GPU Hardware Failure Detection: Proactive detection of GPU hardware issues.


<CollapsibleContent header={<h2><span>Deploying the Solution</span></h2>}>

In this [example](https://github.com/awslabs/ai-on-eks/tree/main/infra/aibrix/terraform), you will provision AIBrix on Amazon EKS.

### Prerequisites

Ensure that you have installed the following tools on your machine.

1. [aws cli](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
2. [kubectl](https://Kubernetes.io/docs/tasks/tools/)
3. [terraform](https://learn.hashicorp.com/tutorials/terraform/install-cli)

### Deploy

**1. Clone the repository:**

```bash
git clone https://github.com/awslabs/ai-on-eks.git
```

:::info
If you are using profile for authentication
set your `export AWS_PROFILE="<PROFILE_name>"` to the desired profile name
:::

**2. Review and customize configurations:**

- Check available addons in `infra/base/terraform/variables.tf`
- Modify addon settings in `infra/aibrix/terraform/blueprint.tfvars` as needed
- Update the AWS region in `blueprint.tfvars`

Navigate into aibrix and run `install.sh` script

```bash
cd ai-on-eks/infra/aibrix
./install.sh
cd ../..
```

</CollapsibleContent>

<CollapsibleContent header={<h3><span>Verify Deployment</span></h3>}>

Update local kubeconfig so we can access kubernetes cluster

:::info
if you haven't set your AWS_REGION, use --region us-east-1 with the below command
:::

```bash
aws eks  update-kubeconfig --name aibrix-on-eks
```

First, let's verify that we have worker nodes running in the cluster.

```bash
kubectl get nodes
```

```bash
NAME                             STATUS   ROLES    AGE   VERSION
ip-100-64-139-184.ec2.internal   Ready    <none>   96m   v1.32.1-eks-5d632ec
ip-100-64-63-169.ec2.internal    Ready    <none>   96m   v1.32.1-eks-5d632ec
```

Next, lets verify all the aibrix pods are running.

``` bash
kubectl get pods -n aibrix-system
```

```bash
NAME                                         READY   STATUS    RESTARTS   AGE
aibrix-controller-manager-5948f8f8b7-pqwjn   1/1     Running   0          83m
aibrix-gateway-plugins-5978d98445-mrgdt      1/1     Running   0          83m
aibrix-gpu-optimizer-64c978ddd8-944mp        1/1     Running   0          83m
aibrix-kuberay-operator-8b65d7cc4-xw6bd      1/1     Running   0          83m
aibrix-metadata-service-5499dc64b7-q6rfc     1/1     Running   0          83m
aibrix-redis-master-576767646c-lqdkk         1/1     Running   0          83m
```

```bash
kubectl get deployments -A
```

```bash
NAMESPACE              NAME                                                 READY   UP-TO-DATE   AVAILABLE   AGE
aibrix-system          aibrix-controller-manager                            1/1     1            1           11m
aibrix-system          aibrix-gateway-plugins                               1/1     1            1           11m
aibrix-system          aibrix-gpu-optimizer                                 1/1     1            1           11m
aibrix-system          aibrix-kuberay-operator                              1/1     1            1           11m
aibrix-system          aibrix-metadata-service                              1/1     1            1           10m
aibrix-system          aibrix-redis-master                                  1/1     1            1           11m
envoy-gateway-system   envoy-aibrix-system-aibrix-eg-903790dc               1/1     1            1           11m
envoy-gateway-system   envoy-gateway                                        1/1     1            1           12m
ingress-nginx          ingress-nginx-controller                             1/1     1            1           11m
karpenter              karpenter                                            2/2     2            2           99m
kube-system            aws-load-balancer-controller                         2/2     2            2           12m
kube-system            coredns                                              2/2     2            2           102m
kube-system            ebs-csi-controller                                   2/2     2            2           80m
kube-system            k8s-neuron-scheduler                                 1/1     1            1           12m
kube-system            my-scheduler                                         1/1     1            1           12m
nvidia-device-plugin   nvidia-device-plugin-node-feature-discovery-master   1/1     1            1           12m
```

:::info

Please refer to [AIBrix Infrastructure](https://awslabs.github.io/ai-on-eks/docs/blueprints) page for deploying AI models on EKS.

:::

</CollapsibleContent>

<CollapsibleContent header={<h3><span>Clean Up</span></h3>}>

:::caution
To avoid unwanted charges to your AWS account, delete all the AWS resources created during this deployment.
:::

This script will cleanup the environment using `-target` option to ensure all the resources are deleted in correct order.

```bash
cd ai-on-eks/infra/aibrix/terraform/_LOCAL
./cleanup.sh
```

</CollapsibleContent>
