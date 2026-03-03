![AI on EKS](website/static/img/aioeks-logo-green.png)
# [AI on Amazon EKS (AIoEKS)](https://awslabs.github.io/ai-on-eks/)
*(Pronounced: "AI on EKS")*
> 💡 **Optimized Solutions for AI and ML on EKS**

> ⚠️ **This repository is under active development as we support the new infrastructure format. Please raise any issues you may encounter**

### Build, Scale, and Optimize AI/ML Platforms on [Amazon EKS](https://aws.amazon.com/eks/) 🚀

Welcome to **AI on EKS**, your gateway to scaling **AI and ML** workloads on Amazon EKS. Unlock the potential of [AI](https://aws.amazon.com/generative-ai/) with a rich collection of Terraform Blueprints featuring best practices for deploying robust solutions with advanced logging and observability.

Explore practical patterns for running AI/ML workloads on EKS, leveraging the power of the [Ray](https://www.ray.io/) ecosystem for distributed computing. Utilize advanced serving solutions like [NVIDIA Triton Server](https://developer.nvidia.com/nvidia-triton-inference-server), [vLLM](https://github.com/vllm-project/vllm) for efficient and scalable model inference, and [TensorRT-LLM](https://developer.nvidia.com/tensorrt) for optimizing deep learning models.

Take advantage of high-performance [NVIDIA GPUs](https://aws.amazon.com/nvidia/) for intensive computational tasks and leverage AWS’s specialized hardware, including [AWS Trainium](https://aws.amazon.com/machine-learning/trainium/) for efficient model training and [AWS Inferentia](https://aws.amazon.com/machine-learning/inferentia/) for cost-effective model inference at scale.

> **Note:** AIoEKS is in active development. For upcoming features and enhancements, check out the [issues](https://github.com/awslabs/ai-on-eks/issues) section.

## 🏃‍♀️Getting Started
In this repository, you'll find a variety of deployment blueprints for creating AI/ML platforms with Amazon EKS clusters. These examples are just a small selection of the available blueprints - visit the [AIoEKS website](https://awslabs.github.io/ai-on-eks/) for the complete list of options.

### 🧠 AI

🚀 [Inference-Ready Cluster](https://awslabs.github.io/ai-on-eks/docs/infra/ai-ml/inference-ready-cluster) 👈 This solution enables supporting multiple inference patters on EKS

🚀 [Inference Charts](https://awslabs.github.io/ai-on-eks/docs/blueprints/inference/inference-charts) 👈 These charts support deploying various models on EKS

🚀 [JARK-Stack on EKS](https://awslabs.github.io/ai-on-eks/docs/infra/ai-ml/jark) 👈 This blueprint deploys JARK stack for AI workloads with NVIDIA GPUs.

🚀 [Generative AI on EKS](https://github.com/awslabs/ai-on-eks/tree/main/blueprints/) 👈 Collection of Generative AI Training and Inference LLM deployment patterns

🚀 [Envoy AI Gateway](blueprints/gateways/envoy-ai-gateway/) 👈 Intelligent routing and management for AI/ML workloads with multi-model routing and rate limiting

## 🎯 Use Cases

🚀 [Multi-Model Routing](blueprints/gateways/envoy-ai-gateway/multi-model-routing/) 👈 Route requests to different AI models based on headers

🚀 [Rate Limiting](blueprints/gateways/envoy-ai-gateway/rate-limiting/) 👈 Usage-based rate limiting with automatic tracking

## 📚 Documentation
For instructions on how to deploy AI on EKS patterns and run sample tests, visit the [AIoEKS website](https://awslabs.github.io/ai-on-eks/).

## 🏆 Motivation
[Kubernetes](https://kubernetes.io/) is a widely adopted system for orchestrating containerized software at scale. As more users migrate their AI and machine learning workloads to Kubernetes, they often face the complexity of managing the Kubernetes ecosystem and selecting the right tools and configurations for their specific needs.

At [AWS](https://aws.amazon.com/), we understand the challenges users encounter when deploying and scaling AI/ML workloads on Kubernetes. To simplify the process and enable users to quickly conduct proof-of-concepts and build clusters, we have developed AI on EKS (AIoEKS). AIoEKS offers opinionated open-source blueprints that provide end-to-end logging and observability, making it easier for users to deploy and manage Ray, vLLM, Kubeflow, MLFlow, Jupyter and other AI/ML workloads. With AIoEKS, users can confidently leverage the power of Kubernetes for their AI and machine learning needs without getting overwhelmed by its complexity.

## 🤝 Support & Feedback
AIoEKS is maintained by AWS Solution Architects and is not an AWS service. Support is provided on a best effort basis by the AI on EKS community. If you have feedback, feature ideas, or wish to report bugs, please use the [Issues](https://github.com/awslabs/ai-on-eks/issues) section of this GitHub.

## 🔐 Security
See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## 💼 License
This library is licensed under the Apache 2.0 License.

## 🙌 Community
We welcome all individuals who are enthusiastic about AI on Kubernetes to become a part of this open source community. Your contributions and participation are invaluable to the success of this project.

Built with ❤️ at AWS.
