# Inference ready Amazon EKS Cluster

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Architecture Steps](#architecture-steps)
- [Plan Your Deployment](#plan-your-deployment)
    - [AWS Services in this Guidance](#aws-services-in-this-guidance)
    - [Cost](#cost)
    - [Sample Cost Table](#sample-cost-table)
- [Security](#security)
- [Quick Start Guide](#quick-start-guide)
    - [Important Setup Instructions](#-important-setup-instructions)
    - [Deploy the Infrastructure](#deploy-the-infrastructure)
        - [Validate the Deployment](#validate-the-deployment)
    - [Deploying Models](#deploying-models)
        - [Prerequisites](#prerequisites)
        - [Create a Hugging Face Token](#how-to-create-a-hugging-face-token)
        - [Create a Cluster Secret](#create-the-cluster-secret)
        - [Deploy the Model](#deploy-a-model)
- [Monitoring and Observability](#-monitoring-and-observability)
- [Troubleshooting](#troubleshooting)
- [Cleanup](#cleanup-the-environment)
- [License](#license)

## Overview

This solution implements a comprehensive, scalable ML inference architecture using Amazon EKS, leveraging both AWS
Neuron processors for cost-effective, accelerated inference, and GPU instances for traditional inference. The system
provides a complete end-to-end platform for deploying large language models and generative AI capabilities along with
observability support.

## Architecture

The architecture diagram illustrates our scalable ML inference solution with the following components:

- **Amazon EKS Cluster**: The foundation of our architecture, providing a managed Kubernetes environment with automated
  provisioning and configuration.

- **Karpenter Auto-scaling**: Dynamically provisions and scales compute resources based on workload demands across
  multiple node pools.

- **Node Pools**:
    - **Neuron-based nodes**: Cost-effective Neuron inference using inf2/trn1 instances
    - **GPU-based nodes**: High-performance inference using NVIDIA GPU instances (g5, g6 families)
    - **x86-based nodes**: General purpose compute for compatibility requirements

- **Model Hosting Services**:
    - **Ray Serve**: Distributed model serving with automatic scaling
    - **Standalone Services**: Direct model deployment for specific use cases
    - **Multi-modal Support**: Text, vision, and reasoning model capabilities
    - **AIBrix**: Distributed KV Caching for resource sharing
    - **LWS**: Multinode distributed inference for very large models

- **Observability & Monitoring**:
    - **Prometheus & Grafana**: Infrastructure monitoring and alerting
    - **Dashboards**: Built-in AI/ML workload specific dashboards

This architecture provides flexibility to choose between cost-optimized inference on Neuron processors or
high-throughput GPU inference based on your specific requirements, all while maintaining elastic scalability through
Kubernetes and Karpenter.

![Architecture Diagram](image/architecture.png)

## Architecture Steps

1) DevOps engineer defines a per-environment Terraform [variable file](terraform/blueprint.tfvars) that controls the
   environment-specific
   configuration.
2) DevOps engineer applies the environment configuration using Terraform following the deployment process defined in the
   guidance.
3) An [Amazon Virtual Private Network (VPC)](https://aws.amazon.com/vpc/) is provisioned and configured based on
   specified configuration. According to best practices for Reliability, 4 Availability zones (AZs) are configured to
   provide the best chance of node acquisition and high availability. Topology awareness defaults to keep AI/ML
   workloads in the same AZ for performance/cost, but is configurable for availability.
4) [Amazon Elastic Kubernetes Service (EKS)](https://aws.amazon.com/eks/) cluster is provisioned with Managed Nodes
   Group that run critical cluster add-ons (CoreDNS, AWS Load Balancer Controller
   and [Karpenter](https://karpenter.sh/)) on its compute node instances. Karpenter will manage compute capacity to
   other add-ons, as well as inference applications that will be deployed by user while prioritizing the most
   cost-effective instances.
5) Other relevant EKS add-ons are deployed based on the configurations defined in the per-environment Terraform
   configuration file.
6) An observability stack including FluentBit and Prometheus is deployed to collect metrics and logs from the
   environment. Service and Pod Monitors are deployed to watch for AI/ML related workloads and collect metrics. Grafana
   and dashboards are deployed to automatically visualize the metrics and logs side by side.
7) Users are now able to deploy AI/ML inference workloads using the AI on EKS inference charts or others.

## Example Model Deployment

![Architecture Diagram](image/deployment.png)

1) ML Engineers use a Helm chart `values` template file to set values for Helm chart for model deployment.
2) ML Engineers use `kubectl/helm` CLI tools to deploy the templated Helm chart to the Amazon EKS environment.
3) Model Helm Templates are applied to Amazon Elastic Kubernetes Service (EKS) API of Inference Ready Amazon EKS cluster
   to start deployment of desired model.
4) Amazon EKS API creates a Kubernetes pod and a service for the single model container deployment.
5) Karpenter auto-scaler provisions Amazon Elastic Compute Cloud (EC2) instances to fulfill the compute node resource
   request to schedule model pods.
6) Docker container image requested by the deployed model for the container is pulled from Amazon Elastic Container
   Registry
7) The weights for the deployed model are pulled from Hugging Face into Amazon S3 for faster loading and loaded into the
   model server
8) External Model consumer users can now port-forward the Kubernetes service port to their local machines for accessing
   the model
9) User/application requests to deployed models in the Inference Ready AWS EKS cluster are routed through the Kubernetes
   service to the model pods, responses are returned the same way.

## Plan your deployment

### AWS services in this Guidance

| **AWS Service**                                                              | **Role**           | **Description**                                                                                             |
|------------------------------------------------------------------------------|--------------------|-------------------------------------------------------------------------------------------------------------|
| [Amazon Elastic Kubernetes Service](https://aws.amazon.com/eks/) ( EKS)      | Core service       | Manages the Kubernetes control plane and worker nodes for container orchestration.                          |
| [Amazon Elastic Compute Cloud](https://aws.amazon.com/ec2/) (EC2)            | Core service       | Provides the compute instances for EKS worker nodes and runs containerized applications.                    |
| [Amazon Virtual Private Cloud](https://aws.amazon.com/vpc/) (VPC)            | Core Service       | Creates an isolated network environment with public and private subnets across multiple Availability Zones. |
| [Amazon Elastic Container Registry](http://aws.amazon.com/ecr/) (ECR)        | Supporting service | Stores and manages Docker container images for EKS deployments.                                             |
| [Elastic Load Balancing](https://aws.amazon.com/elasticloadbalancing/) (NLB) | Supporting service | Distributes incoming traffic across multiple targets in the EKS cluster.                                    |
| [Amazon Elastic Block Store](https://aws.amazon.com/ebs) (EBS)               | Supporting service | Provides persistent block storage volumes for EC2 instances in the EKS cluster.                             |
| [AWS Key Management Service](https://aws.amazon.com/kms/) (KMS)              | Security service   | Manages encryption keys for securing data in EKS and other AWS services.                                    |

### Cost

You are responsible for the cost of the AWS services used while running this guidance.
As of August 2025, the cost for running this guidance with the default settings in the US West (Oregon) Region is
approximately **$296.21/month**.

We recommend creating a [budget](https://alpha-docs-aws.amazon.com/awsaccountbilling/latest/aboutv2/budgets-create.html)
through [AWS Cost Explorer](http://aws.amazon.com/aws-cost-management/aws-cost-explorer/) to help manage costs. Prices
are subject to change. For full details, refer to the pricing webpage for each AWS service used in this guidance.

### Sample cost table

The following table provides a sample cost breakdown for deploying this guidance with the default parameters in the
`us-west-2` (Oregon) Region for one month. This estimate is based on the AWS Pricing Calculator output for the full
deployment as per the guidance. This **does not** factor any model deployments on top of the running environment.

| **AWS service**                  | Dimensions                        | Cost, month [USD] |
|----------------------------------|-----------------------------------|-------------------|
| Amazon EKS                       | 1 cluster                         | $73.00            |
| Amazon VPC                       | 1 NAT Gateways                    | $33.75            |
| Amazon EC2                       | 2 m5.large instances              | $156.16           |
| Amazon EBS                       | gp3 storage volumes and snapshots | $7.20             |
| Elastic Load Balancer            | 1 NLB for workloads               | $16.46            |
| Amazon VPC                       | Public IP addresses               | $3.65             |
| AWS Key Management Service (KMS) | Keys and requests                 | $6.00             |
| **TOTAL**                        |                                   | **$296.21/month** |

For a more accurate estimate based on your specific configuration and usage patterns, we recommend using
the [AWS Pricing Calculator](https://calculator.aws).

## Security

When you build systems on AWS infrastructure, security responsibilities are shared between you and AWS.
This [shared responsibility model](https://aws.amazon.com/compliance/shared-responsibility-model/) reduces your
operational burden because AWS operates, manages, and controls the components including the host operating system, the
virtualization layer, and the physical security of the facilities in which the services operate. For more information
about AWS security, visit [AWS Cloud Security](http://aws.amazon.com/security/).

This guidance implements several security best practices and AWS services to enhance the security posture of your EKS
Workload Ready Cluster. Here are the key security components and considerations:

### Identity and Access Management (IAM)

- **EKS Managed Node Groups**: These use IAM roles with specific permissions required for nodes to join the cluster and
  for pods to access AWS services.

### Network Security

- **Amazon VPC**: The EKS cluster is deployed within a custom VPC with public and private subnets across multiple
  Availability Zones, providing network isolation.
- **Security Groups**: Although not explicitly shown in the diagram, security groups are typically used to control
  inbound and outbound traffic to EC2 instances and other resources within the VPC.
- **NAT Gateways**: Deployed in public subnets to allow outbound internet access for resources in private subnets while
  preventing inbound access from the internet.

### Data Protection

- **Amazon EBS Encryption**: EBS volumes used by EC2 instances are typically encrypted to protect data at rest.
- **AWS Key Management Service (KMS)**: Used for managing encryption keys for various services, including EBS volume
  encryption.

### Kubernetes-specific Security

- **Kubernetes RBAC**: Role-Based Access Control is implemented within the EKS cluster to manage fine-grained access to
  Kubernetes resources.

### Secrets Management

- **AWS Secrets Manager**: While not explicitly shown in the diagram, it's commonly used to securely store and manage
  sensitive information such as database credentials, API keys, and other secrets used by applications running on EKS.

### Additional Security Considerations

- Regularly update and patch EKS clusters, worker nodes, and container images.
- Implement network policies to control pod-to-pod communication within the cluster.
- Use Pod Security Policies or Pod Security Standards to enforce security best practices for pods.
- Implement proper logging and auditing mechanisms for both AWS and Kubernetes resources.
- Regularly review and rotate IAM and Kubernetes RBAC permissions.

## Quick Start Guide

The solution comes in two parts:

- The infrastructure for running inference workloads (this)
- The models that can be deployed on top of a running environment (
  the [inference charts](https://github.com/awslabs/ai-on-eks-charts))

### ⚠️ Important Setup Instructions

**Before proceeding with this solution, ensure you have:**

- **AWS CLI configured** with appropriate permissions for EKS, ECR, CloudFormation, and other AWS services
- **kubectl installed** and configured to access your target AWS region
- **Sufficient AWS service quotas** - This solution requires multiple EC2 instances, EKS cluster, and other AWS
  resources

**Recommended Setup Verification:**

```bash
# Verify AWS CLI access
aws sts get-caller-identity

# Verify kubectl installation
kubectl version --client

# Check available AWS regions and quotas
aws ec2 describe-regions
aws service-quotas get-service-quota --service-code ec2 --quota-code L-1216C47A
```

**Cost Awareness:** This solution will incur AWS charges. Review the cost breakdown section below and set up billing
alerts before deployment.

### Deploy the Infrastructure

The following is a quick way to deploy the infrastructure. It will create everything and return the command to configure
`kubectl` for this cluster. Note, it will take about 15 minutes to run.

```bash
git clone https://github.com/awslabs/ai-on-eks.git
cd ai-on-eks
cd infra/solutions/inference-ready-cluster
./install.sh
```

#### Validate the Deployment

To validate that everything deployed properly, you can run

```bash
kubectl get svc,pod,deployment -A
```

You should see the following output (expand the section to see the output)

<details>

```text
    NAMESPACE              NAME                                                             TYPE           CLUSTER-IP       EXTERNAL-IP                                                                     PORT(S)                                        AGE
    aibrix-system          service/aibrix-controller-manager-metrics-service                ClusterIP      172.20.218.39    <none>                                                                          8080/TCP                                       13d
    aibrix-system          service/aibrix-gateway-plugins                                   ClusterIP      172.20.142.245   <none>                                                                          50052/TCP                                      13d
    aibrix-system          service/aibrix-gpu-optimizer                                     ClusterIP      172.20.14.220    <none>                                                                          8080/TCP                                       13d
    aibrix-system          service/aibrix-kuberay-operator                                  ClusterIP      172.20.240.255   <none>                                                                          8080/TCP                                       13d
    aibrix-system          service/aibrix-metadata-service                                  ClusterIP      172.20.252.24    <none>                                                                          8090/TCP                                       13d
    aibrix-system          service/aibrix-redis-master                                      ClusterIP      172.20.155.43    <none>                                                                          6379/TCP                                       13d
    argocd                 service/argocd-applicationset-controller                         ClusterIP      172.20.139.94    <none>                                                                          7000/TCP                                       13d
    argocd                 service/argocd-dex-server                                        ClusterIP      172.20.127.60    <none>                                                                          5556/TCP,5557/TCP                              13d
    argocd                 service/argocd-redis                                             ClusterIP      172.20.48.202    <none>                                                                          6379/TCP                                       13d
    argocd                 service/argocd-repo-server                                       ClusterIP      172.20.232.147   <none>                                                                          8081/TCP                                       13d
    argocd                 service/argocd-server                                            ClusterIP      172.20.233.191   <none>                                                                          80/TCP,443/TCP                                 13d
    default                service/etcd-client                                              ClusterIP      172.20.47.224    <none>                                                                          2379/TCP                                       12d
    default                service/etcd-server                                              ClusterIP      172.20.69.95     <none>                                                                          2379/TCP,2380/TCP                              12d
    default                service/kubernetes                                               ClusterIP      172.20.0.1       <none>                                                                          443/TCP                                        13d
    envoy-gateway-system   service/envoy-aibrix-system-aibrix-eg-903790dc                   ClusterIP      172.20.249.100   <none>                                                                          80/TCP                                         13d
    envoy-gateway-system   service/envoy-gateway                                            ClusterIP      172.20.113.229   <none>                                                                          18000/TCP,18001/TCP,18002/TCP,19001/TCP        13d
    ingress-nginx          service/ingress-nginx-controller                                 LoadBalancer   172.20.27.209    k8s-ingressn-ingressn-ffa534dcb1-b4b54bcc24eaeddd.elb.us-west-2.amazonaws.com   80:31646/TCP,443:32024/TCP                     13d
    ingress-nginx          service/ingress-nginx-controller-admission                       ClusterIP      172.20.249.118   <none>                                                                          443/TCP                                        13d
    karpenter              service/karpenter                                                ClusterIP      172.20.149.70    <none>                                                                          8080/TCP                                       13d
    kube-system            service/aws-load-balancer-webhook-service                        ClusterIP      172.20.83.104    <none>                                                                          443/TCP                                        13d
    kube-system            service/eks-extension-metrics-api                                ClusterIP      172.20.87.142    <none>                                                                          443/TCP                                        13d
    kube-system            service/k8s-neuron-scheduler                                     ClusterIP      172.20.248.128   <none>                                                                          12345/TCP                                      13d
    kube-system            service/kube-dns                                                 ClusterIP      172.20.0.10      <none>                                                                          53/UDP,53/TCP,9153/TCP                         13d
    kube-system            service/kube-prometheus-stack-kubelet                            ClusterIP      None             <none>                                                                          10250/TCP,10255/TCP,4194/TCP                   13d
    kuberay-operator       service/kuberay-operator                                         ClusterIP      172.20.117.159   <none>                                                                          8080/TCP                                       13d
    lws-system             service/lws-controller-manager-metrics-service                   ClusterIP      172.20.17.186    <none>                                                                          8443/TCP                                       13d
    lws-system             service/lws-webhook-service                                      ClusterIP      172.20.173.201   <none>                                                                          443/TCP                                        13d
    monitoring             service/alertmanager-operated                                    ClusterIP      None             <none>                                                                          9093/TCP,9094/TCP,9094/UDP                     13d
    monitoring             service/dcgm-exporter                                            ClusterIP      172.20.79.5      <none>                                                                          9400/TCP                                       13d
    monitoring             service/fluent-bit                                               ClusterIP      172.20.111.213   <none>                                                                          2020/TCP                                       13d
    monitoring             service/kube-prometheus-stack-alertmanager                       ClusterIP      172.20.45.163    <none>                                                                          9093/TCP,8080/TCP                              13d
    monitoring             service/kube-prometheus-stack-coredns                            ClusterIP      None             <none>                                                                          9153/TCP                                       13d
    monitoring             service/kube-prometheus-stack-grafana                            ClusterIP      172.20.251.144   <none>                                                                          80/TCP                                         13d
    monitoring             service/kube-prometheus-stack-kube-controller-manager            ClusterIP      None             <none>                                                                          10257/TCP                                      13d
    monitoring             service/kube-prometheus-stack-kube-etcd                          ClusterIP      None             <none>                                                                          2381/TCP                                       13d
    monitoring             service/kube-prometheus-stack-kube-proxy                         ClusterIP      None             <none>                                                                          10249/TCP                                      13d
    monitoring             service/kube-prometheus-stack-kube-scheduler                     ClusterIP      None             <none>                                                                          10259/TCP                                      13d
    monitoring             service/kube-prometheus-stack-kube-state-metrics                 ClusterIP      172.20.81.57     <none>                                                                          8080/TCP                                       13d
    monitoring             service/kube-prometheus-stack-operator                           ClusterIP      172.20.163.90    <none>                                                                          443/TCP                                        13d
    monitoring             service/kube-prometheus-stack-prometheus                         ClusterIP      172.20.1.251     <none>                                                                          9090/TCP,8080/TCP                              13d
    monitoring             service/kube-prometheus-stack-prometheus-node-exporter           ClusterIP      172.20.88.160    <none>                                                                          9100/TCP                                       13d
    monitoring             service/my-cluster                                               ClusterIP      172.20.54.44     <none>                                                                          9200/TCP,9300/TCP,9600/TCP,9650/TCP            13d
    monitoring             service/my-cluster-dashboards                                    ClusterIP      172.20.161.35    <none>                                                                          5601/TCP                                       13d
    monitoring             service/my-cluster-masters                                       ClusterIP      None             <none>                                                                          9200/TCP,9300/TCP                              13d
    monitoring             service/opencost                                                 ClusterIP      172.20.162.78    <none>                                                                          9003/TCP,9090/TCP                              13d
    monitoring             service/opensearch-discovery                                     ClusterIP      None             <none>                                                                          9300/TCP                                       13d
    monitoring             service/opensearch-operator-controller-manager-metrics-service   ClusterIP      172.20.183.236   <none>                                                                          8443/TCP                                       13d
    monitoring             service/prometheus-operated                                      ClusterIP      None             <none>                                                                          9090/TCP                                       13d

    NAMESPACE              NAME                                                                  READY   STATUS      RESTARTS        AGE
    aibrix-system          pod/aibrix-controller-manager-5948f8f8b7-qjm7z                        1/1     Running     0               13d
    aibrix-system          pod/aibrix-gateway-plugins-5978d98445-qj2jw                           1/1     Running     0               13d
    aibrix-system          pod/aibrix-gpu-optimizer-64c978ddd8-bw7hk                             1/1     Running     0               13d
    aibrix-system          pod/aibrix-kuberay-operator-8b65d7cc4-xrcm6                           1/1     Running     0               13d
    aibrix-system          pod/aibrix-metadata-service-5499dc64b7-69tzc                          1/1     Running     0               13d
    aibrix-system          pod/aibrix-redis-master-576767646c-w9lhl                              1/1     Running     0               13d
    argocd                 pod/argocd-application-controller-0                                   1/1     Running     0               13d
    argocd                 pod/argocd-applicationset-controller-6847f76cb8-svwvt                 1/1     Running     0               13d
    argocd                 pod/argocd-dex-server-f6d74975f-g5rj8                                 1/1     Running     0               13d
    argocd                 pod/argocd-notifications-controller-86f4bb887d-sgxlb                  1/1     Running     0               13d
    argocd                 pod/argocd-redis-588f9bcd4d-9tncd                                     1/1     Running     0               13d
    argocd                 pod/argocd-repo-server-5cbcc778f4-kd4ll                               1/1     Running     0               13d
    argocd                 pod/argocd-server-7c9898bc58-vfqwn                                    1/1     Running     0               13d
    envoy-gateway-system   pod/envoy-aibrix-system-aibrix-eg-903790dc-567ff75b87-22ctt           2/2     Running     0               13d
    envoy-gateway-system   pod/envoy-gateway-6d7859b6bf-6hhf5                                    1/1     Running     0               13d
    ingress-nginx          pod/ingress-nginx-controller-58f4c5584-wt6rk                          1/1     Running     0               13d
    karpenter              pod/karpenter-849fd44788-4fgml                                        1/1     Running     0               13d
    karpenter              pod/karpenter-849fd44788-zbm9z                                        1/1     Running     0               13d
    kube-system            pod/aws-load-balancer-controller-c495bf799-crnlh                      1/1     Running     0               13d
    kube-system            pod/aws-load-balancer-controller-c495bf799-nwkqv                      1/1     Running     0               13d
    kube-system            pod/aws-node-6ff6l                                                    2/2     Running     0               8d
    kube-system            pod/aws-node-728vt                                                    2/2     Running     0               2d16h
    kube-system            pod/aws-node-87jfl                                                    2/2     Running     0               13d
    kube-system            pod/aws-node-wtnlj                                                    2/2     Running     0               13d
    kube-system            pod/aws-node-zzc4g                                                    2/2     Running     0               2d16h
    kube-system            pod/coredns-7bf648ff5d-98bp4                                          1/1     Running     0               13d
    kube-system            pod/coredns-7bf648ff5d-w56nm                                          1/1     Running     0               13d
    kube-system            pod/ebs-csi-controller-5bdc7bfdb6-79658                               6/6     Running     0               13d
    kube-system            pod/ebs-csi-controller-5bdc7bfdb6-958zf                               6/6     Running     0               13d
    kube-system            pod/ebs-csi-node-4z2mb                                                3/3     Running     0               13d
    kube-system            pod/ebs-csi-node-8qq2s                                                3/3     Running     0               2d16h
    kube-system            pod/ebs-csi-node-q9h5r                                                3/3     Running     0               2d16h
    kube-system            pod/ebs-csi-node-t77j9                                                3/3     Running     0               13d
    kube-system            pod/ebs-csi-node-w9mh8                                                3/3     Running     0               8d
    kube-system            pod/eks-pod-identity-agent-jjfz4                                      1/1     Running     0               13d
    kube-system            pod/eks-pod-identity-agent-jthdk                                      1/1     Running     0               2d16h
    kube-system            pod/eks-pod-identity-agent-ng556                                      1/1     Running     0               8d
    kube-system            pod/eks-pod-identity-agent-q6ths                                      1/1     Running     0               2d16h
    kube-system            pod/eks-pod-identity-agent-rwkr9                                      1/1     Running     0               13d
    kube-system            pod/k8s-neuron-scheduler-56f6c8bd67-hbzgz                             1/1     Running     0               13d
    kube-system            pod/kube-proxy-4wf7s                                                  1/1     Running     0               2d16h
    kube-system            pod/kube-proxy-7dm2x                                                  1/1     Running     0               2d16h
    kube-system            pod/kube-proxy-9d9cm                                                  1/1     Running     0               8d
    kube-system            pod/kube-proxy-lt4sp                                                  1/1     Running     0               13d
    kube-system            pod/kube-proxy-nklwj                                                  1/1     Running     0               13d
    kube-system            pod/my-scheduler-6959876cb4-gprm5                                     1/1     Running     0               13d
    kuberay-operator       pod/kuberay-operator-6d988d7dd9-ncx4h                                 1/1     Running     0               13d
    lws-system             pod/lws-controller-manager-cbb85458b-7cvhr                            1/1     Running     0               13d
    lws-system             pod/lws-controller-manager-cbb85458b-dqj8g                            1/1     Running     0               13d
    monitoring             pod/alertmanager-kube-prometheus-stack-alertmanager-0                 2/2     Running     0               13d
    monitoring             pod/fluent-bit-52m29                                                  1/1     Running     0               2d16h
    monitoring             pod/fluent-bit-hb824                                                  1/1     Running     0               13d
    monitoring             pod/fluent-bit-hsptw                                                  1/1     Running     0               13d
    monitoring             pod/fluent-bit-nkmrq                                                  1/1     Running     0               8d
    monitoring             pod/fluent-bit-qdsm2                                                  1/1     Running     0               2d16h
    monitoring             pod/fluent-operator-7f75b8ccf4-z5924                                  1/1     Running     0               13d
    monitoring             pod/kube-prometheus-stack-grafana-c64f79c4f-zqlm7                     3/3     Running     0               13d
    monitoring             pod/kube-prometheus-stack-kube-state-metrics-77976dc6c4-fff28         1/1     Running     0               13d
    monitoring             pod/kube-prometheus-stack-operator-6655669d75-4kh9s                   1/1     Running     0               13d
    monitoring             pod/kube-prometheus-stack-prometheus-node-exporter-7xbrs              1/1     Running     0               13d
    monitoring             pod/kube-prometheus-stack-prometheus-node-exporter-gwkb2              1/1     Running     0               13d
    monitoring             pod/kube-prometheus-stack-prometheus-node-exporter-k6zl7              1/1     Running     0               2d16h
    monitoring             pod/kube-prometheus-stack-prometheus-node-exporter-pl6m7              1/1     Running     0               2d16h
    monitoring             pod/kube-prometheus-stack-prometheus-node-exporter-st9kp              1/1     Running     0               8d
    monitoring             pod/opencost-bd64bfbf5-jbfvr                                          2/2     Running     0               13d
    monitoring             pod/opensearch-dashboards-84675f8b9-6jd2h                             1/1     Running     0               13d
    monitoring             pod/opensearch-dashboards-84675f8b9-9mcs2                             1/1     Running     0               13d
    monitoring             pod/opensearch-masters-0                                              1/1     Running     0               13d
    monitoring             pod/opensearch-masters-1                                              1/1     Running     0               8d
    monitoring             pod/opensearch-masters-2                                              1/1     Running     0               8d
    monitoring             pod/opensearch-operator-controller-manager-58b76955b9-w46gl           2/2     Running     0               13d
    monitoring             pod/opensearch-securityconfig-update-4fdcz                            0/1     Completed   0               13d
    monitoring             pod/prometheus-kube-prometheus-stack-prometheus-0                     2/2     Running     0               13d
    nvidia-device-plugin   pod/nvidia-device-plugin-node-feature-discovery-master-77b96ddp8h25   1/1     Running     0               13d

    NAMESPACE              NAME                                                                 READY   UP-TO-DATE   AVAILABLE   AGE
    aibrix-system          deployment.apps/aibrix-controller-manager                            1/1     1            1           13d
    aibrix-system          deployment.apps/aibrix-gateway-plugins                               1/1     1            1           13d
    aibrix-system          deployment.apps/aibrix-gpu-optimizer                                 1/1     1            1           13d
    aibrix-system          deployment.apps/aibrix-kuberay-operator                              1/1     1            1           13d
    aibrix-system          deployment.apps/aibrix-metadata-service                              1/1     1            1           13d
    aibrix-system          deployment.apps/aibrix-redis-master                                  1/1     1            1           13d
    argocd                 deployment.apps/argocd-applicationset-controller                     1/1     1            1           13d
    argocd                 deployment.apps/argocd-dex-server                                    1/1     1            1           13d
    argocd                 deployment.apps/argocd-notifications-controller                      1/1     1            1           13d
    argocd                 deployment.apps/argocd-redis                                         1/1     1            1           13d
    argocd                 deployment.apps/argocd-repo-server                                   1/1     1            1           13d
    argocd                 deployment.apps/argocd-server                                        1/1     1            1           13d
    envoy-gateway-system   deployment.apps/envoy-aibrix-system-aibrix-eg-903790dc               1/1     1            1           13d
    envoy-gateway-system   deployment.apps/envoy-gateway                                        1/1     1            1           13d
    ingress-nginx          deployment.apps/ingress-nginx-controller                             1/1     1            1           13d
    karpenter              deployment.apps/karpenter                                            2/2     2            2           13d
    kube-system            deployment.apps/aws-load-balancer-controller                         2/2     2            2           13d
    kube-system            deployment.apps/coredns                                              2/2     2            2           13d
    kube-system            deployment.apps/ebs-csi-controller                                   2/2     2            2           13d
    kube-system            deployment.apps/k8s-neuron-scheduler                                 1/1     1            1           13d
    kube-system            deployment.apps/my-scheduler                                         1/1     1            1           13d
    kuberay-operator       deployment.apps/kuberay-operator                                     1/1     1            1           13d
    lws-system             deployment.apps/lws-controller-manager                               2/2     2            2           13d
    monitoring             deployment.apps/fluent-operator                                      1/1     1            1           13d
    monitoring             deployment.apps/kube-prometheus-stack-grafana                        1/1     1            1           13d
    monitoring             deployment.apps/kube-prometheus-stack-kube-state-metrics             1/1     1            1           13d
    monitoring             deployment.apps/kube-prometheus-stack-operator                       1/1     1            1           13d
    monitoring             deployment.apps/opencost                                             1/1     1            1           13d
    monitoring             deployment.apps/opensearch-dashboards                                2/2     2            2           13d
    monitoring             deployment.apps/opensearch-operator-controller-manager               1/1     1            1           13d
    nvidia-device-plugin   deployment.apps/nvidia-device-plugin-node-feature-discovery-master   1/1     1            1           13d
```

</details>

### Deploying models

#### Prerequisites

- EKS cluster set up following the steps above
- `kubectl` configured to access your cluster
- a Hugging Face Token
- a configured secret from the Hugging Face Token

#### How to create a Hugging Face Token

To access Hugging Face models, you'll need to create an access token:

1. **Sign up or log in** to [Hugging Face](https://huggingface.co/)
2. **Navigate to Settings**: Click on your profile picture in the top right corner and select "Settings"
3. **Access Tokens**: In the left sidebar, click on "Access Tokens"
4. **Create New Token**: Click "New token" button
5. **Configure Token**:
    - **Name**: Give your token a descriptive name (e.g., "EKS-ML-Inference")
    - **Type**: Select "Read" for most use cases (allows downloading models)
    - **Repositories**: Leave empty to access all public repositories, or specify particular ones
6. **Generate Token**: Click "Generate a token"
7. **Copy and Store**: Copy the generated token immediately and store it securely

**Important Notes**:

- Keep your token secure and never share it publicly
- You can revoke tokens at any time from the same settings page
- Consider using organization tokens with appropriate permissions
- Some models may require additional permissions or agreements before access

#### Create the cluster secret

Replace `your_huggingface_token` with the token from the previous step

```bash
kubectl create secret generic hf-token --from-literal=token=your_huggingface_token
```

#### Deploy a model

The following will deploy a Qwen 3 1.7B model on a GPU node.

```bash
helm repo add ai-on-eks https://awslabs.github.io/ai-on-eks-charts/
helm repo update

helm install qwen3-1-7b ai-on-eks/inference-charts -f https://raw.githubusercontent.com/awslabs/ai-on-eks-charts/refs/heads/main/charts/inference-charts/values-qwen3-1.7b-vllm.yaml
```

Please take a look at all the different deployment options in
the [inference charts readme](https://github.com/awslabs/ai-on-eks-charts/blob/main/charts/inference-charts/README.md).

### 📊 Monitoring and Observability

The solution includes comprehensive observability features:

- **Prometheus Integration**: Enables automated metric collection of system and AI workloads.
- **Fluent Bit Log Aggregation**: Automates log collection for system and AI workloads.
- **OpenSearch Log Backend**: Robust, 3 replica stateful log storage
- **Grafana Dashboards**: Out of the box dashboards for aggregating logs and metrics for AI inference
- **Alertmanager**: Supports automated alerting on metrics

#### Connect

```bash
kubectl port-forward -n monitoring svc/kube-prometheus-stack-grafana 3000:80
```

You can now visit http://localhost:3000 and log in with username: `admin`, password: `prom-operator` to access Grafana.

The solution includes an inference dashboard
available [here](http://localhost:3000/d/bec31e71-3ac5-4133-b2e3-b9f75c8ab56c/inference-dashboard?orgId=1&refresh=5s).

## Troubleshooting

This section covers common issues you may encounter when deploying and operating the inference-ready EKS cluster, along
with detailed solutions and diagnostic steps.

### Deployment Issues

#### 1. Terraform Apply Failures

**Symptoms:**

- Terraform fails during `terraform apply` with resource creation errors
- Module-specific failures during sequential deployment

**Common Causes & Solutions:**

**Insufficient AWS Permissions:**

```bash
# Verify your AWS credentials and permissions
aws sts get-caller-identity
aws iam get-user

# Required permissions include:
# - EKS cluster creation and management
# - EC2 instance management
# - VPC and networking resources
# - IAM role creation and attachment
# - KMS key management
```

**Service Quota Limits:**

```bash
# Check EC2 service quotas
aws service-quotas get-service-quota --service-code ec2 --quota-code L-1216C47A  # Running On-Demand instances
aws service-quotas get-service-quota --service-code ec2 --quota-code L-34B43A08  # Running On-Demand G instances
aws service-quotas get-service-quota --service-code ec2 --quota-code L-6E869C2A  # Running On-Demand Inf instances

# Request quota increases if needed
aws service-quotas request-service-quota-increase --service-code ec2 --quota-code L-34B43A08 --desired-value 32
```

**Region Availability:**

```bash
# Verify instance types are available in your region
aws ec2 describe-instance-type-offerings --location-type availability-zone --filters Name=instance-type,Values=g5.xlarge,inf2.xlarge
```

#### 2. EKS Cluster Creation Issues

**Symptoms:**

- EKS cluster fails to create or becomes stuck in "CREATING" state
- Node groups fail to join the cluster

**Diagnostic Steps:**

```bash
# Check cluster status
aws eks describe-cluster --name inference-cluster --region us-west-2
```

**Common Solutions:**

- Ensure VPC has sufficient IP addresses across 4 availability zones
- Verify NAT Gateway creation in public subnets
- Check security group configurations allow required EKS communication

### Node and Pod Issues

#### 3. Pods Stuck in Pending State

**Symptoms:**

- Inference workloads remain in "Pending" status
- Karpenter not provisioning nodes

**Diagnostic Commands:**

```bash
# Check pod events and resource requests
kubectl describe pod <pod-name> -n <namespace>

# Check Karpenter logs
kubectl logs -n karpenter deployment/karpenter

# Check available nodes and their capacity
kubectl get nodes -o wide
kubectl describe nodes
```

**Common Causes & Solutions:**

**Insufficient GPU/Neuron Quotas:**

```bash
# Verify Karpenter NodePool configurations
kubectl get nodepool -o yaml
```

#### 4. GPU Detection and Device Plugin Issues

**Symptoms:**

- GPU nodes show 0 allocatable GPUs
- NVIDIA device plugin not running

**Diagnostic Steps:**

```bash
# Verify GPU visibility on nodes
kubectl get nodes -o json | jq '.items[] | {name: .metadata.name, gpus: .status.allocatable["nvidia.com/gpu"]}'

# Check node labels
kubectl get nodes --show-labels | grep gpu
```

**Solutions:**

```bash
# Restart NVIDIA device plugin if needed
kubectl delete pods -n nvidia-device-plugin -l app.kubernetes.io/name=nvidia-device-plugin
```

#### 5. AWS Neuron Setup Issues

**Symptoms:**

- Neuron devices not detected on inf2/trn1 instances
- Neuron device plugin failing

**Diagnostic Commands:**

```bash
# Check Neuron device plugin
kubectl get pods -n kube-system | grep neuron

# Verify Neuron devices
kubectl get nodes -o json | jq '.items[] | {name: .metadata.name, neuron: .status.allocatable["aws.amazon.com/neuron"]}'

# Check Neuron scheduler
kubectl get pods -n kube-system | grep my-scheduler
```

**Solutions:**

```bash
# Verify Neuron runtime installation
kubectl describe node <inf2-node> | grep neuron

# Check Neuron device plugin logs
kubectl logs -n kube-system <neuron-device-plugin-pod>
```

### Model Deployment Issues

#### 6. Model Download Failures

**Symptoms:**

- Pods fail to start with image pull or model download errors
- Hugging Face authentication failures

**Diagnostic Steps:**

```bash
# Check pod logs for download errors
kubectl logs <pod-name> -n <namespace>

# Verify Hugging Face token secret
kubectl get secret hf-token -o yaml
kubectl get secret hf-token -o jsonpath='{.data.token}' | base64 -d
```

**Solutions:**

```bash
# Recreate Hugging Face token secret
kubectl delete secret hf-token
kubectl create secret generic hf-token --from-literal=token=<your-hf-token>

# Check internet connectivity from pods
kubectl run test-pod --image=curlimages/curl -it --rm -- curl -I https://huggingface.co
```

#### 7. Out of Memory (OOM) Issues

**Symptoms:**

- Pods getting killed with OOMKilled status
- Models failing to load completely

**Diagnostic Commands:**

```bash
# Check pod resource usage
kubectl top pods -n <namespace>

# Check node memory usage
kubectl top nodes

# Review pod events for OOM kills
kubectl get events --field-selector reason=OOMKilling
```

**Solutions:**

```bash
# Increase instance type to get larger GPU
# Consider using larger instance types or model sharding
```

### Networking and Load Balancer Issues

#### 8. Service Connectivity Problems

**Symptoms:**

- Cannot access inference endpoints
- Load balancer not provisioning

**Diagnostic Steps:**

```bash
# Check service status
kubectl get svc -A

# Check AWS Load Balancer Controller
kubectl logs -n kube-system deployment/aws-load-balancer-controller

# Verify security groups and NACLs
aws ec2 describe-security-groups --filters "Name=group-name,Values=*inference-cluster*"
```

**Solutions:**

```bash
# Restart AWS Load Balancer Controller
kubectl rollout restart deployment/aws-load-balancer-controller -n kube-system

# Check ingress annotations and configurations
kubectl describe ingress <ingress-name>
```

### Monitoring and Observability Issues

#### 9. Prometheus/Grafana Not Working

**Symptoms:**

- Monitoring dashboards not accessible
- Metrics not being collected

**Diagnostic Commands:**

```bash
# Check monitoring stack pods
kubectl get pods -n monitoring

# Check Prometheus targets
kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090
# Navigate to http://localhost:9090/targets

# Check Grafana access
kubectl port-forward -n monitoring svc/kube-prometheus-stack-grafana 3000:80
# Navigate to http://localhost:3000
```

**Solutions:**

```bash
# Restart monitoring components
kubectl rollout restart deployment/kube-prometheus-stack-grafana -n monitoring

# Check service monitors
kubectl get servicemonitor -A
```

### Performance and Scaling Issues

#### 10. Slow Model Inference

**Symptoms:**

- High latency in model responses
- Poor throughput performance

**Diagnostic Steps:**

```bash
# Check resource utilization
kubectl top pods -n <namespace> --containers

# Monitor GPU utilization (if using GPUs)
kubectl exec -it <pod-name> -- nvidia-smi
```

**Solutions:**

- Verify model is using appropriate hardware acceleration
- Check if multiple models are competing for resources
- Optimize model parameters for latency
- Scale up model and use load balancing

### General Debugging Commands

```bash
# Get cluster information
kubectl cluster-info
kubectl get nodes -o wide

# Check all system pods
kubectl get pods -A | grep -v Running

# View recent events
kubectl get events --sort-by='.lastTimestamp' -A

# Check Karpenter provisioning
kubectl logs -n karpenter deployment/karpenter --tail=100

# Verify EKS add-ons
aws eks describe-addon --cluster-name inference-cluster --addon-name vpc-cni
```

### Getting Additional Help

If you continue to experience issues:

1. **Check AWS Service Health**: Visit the [AWS Service Health Dashboard](https://status.aws.amazon.com/)
2. **Review CloudWatch Logs**: Check EKS control plane logs in CloudWatch
3. **Consult Documentation**: Refer to
   the [EKS Troubleshooting Guide](https://docs.aws.amazon.com/eks/latest/userguide/troubleshooting.html)
4. **Community Support**: Post questions in the [AI on EKS GitHub Issues](https://github.com/awslabs/ai-on-eks/issues)

## Cleanup the Environment

When you are done using the environment, you can delete all its resources by running the following command (assuming the
root of the git repository):

```bash
cd infra/solutions/inference-ready-cluster/terraform/_LOCAL
./cleanup.sh
```

This cleanup script will remove the EKS environment and VPC and anything contained in the VPC that was created by the
installation script. Note, it will not remove anything that was created in S3 or stored outside the components that were
directly created by the deployment. You will need to remove them yourself to not incur any further potential costs.

## License

This solution is licensed under the Apache-2.0 License, please find the [License here](../../../LICENSE)
