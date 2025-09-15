---
sidebar_label: AIBrix on EKS
---
import CollapsibleContent from '../../../../src/components/CollapsibleContent';


# AIBrix

AIBrix is an open source initiative designed to provide essential building blocks to construct scalable GenAI inference infrastructure. AIBrix delivers a cloud-native solution optimized for deploying, managing, and scaling large language model (LLM) inference, tailored specifically to enterprise needs.
![Alt text](https://aibrix.readthedocs.io/latest/_images/aibrix-architecture-v1.jpeg)

### Features
* LLM Gateway and Routing: Efficiently manage and direct traffic across multiple models and replicas.
* High-Density LoRA Management: Streamlined support for lightweight, low-rank adaptations of models.
* Distributed Inference: Scalable architecture to handle large workloads across multiple nodes.
* LLM App-Tailored Autoscaler: Dynamically scale inference resources based on real-time demand.
* Unified AI Runtime: A versatile sidecar enabling metric standardization, model downloading, and management.
* Heterogeneous-GPU Inference: Cost-effective SLO-driven LLM inference using heterogeneous GPUs.
* GPU Hardware Failure Detection: Proactive detection of GPU hardware issues.


<CollapsibleContent header={<h2><span>Deploying the Solution</span></h2>}>

:::warning
Before deploying this blueprint, it is important to be cognizant of the costs associated with the utilization of GPU Instances.
:::

Please refer to [AI](https://awslabs.github.io/ai-on-eks/docs/infra/aibrix) page for deploying AIBrix models on EKS.

</CollapsibleContent>


### Checking AIBrix Installation

Please run the below commands to check the AIBrix installation

``` bash
kubectl get pods -n aibrix-system
```

Wait till all the pods are in Running status.

#### Running a model on AiBrix system

We will now run Deepseek-Distill-llama-8b model using AIBrix on EKS.

Please run the below command.

```bash
kubectl apply -f blueprints/inference/aibrix/deepseek-distill.yaml
```

This will deploy the model on deepseek-aibrix namespace. Wait for few minutes and run

```bash
kubectl get pods -n deepseek-aibrix
```
Wait for the pod to be in running state.

#### Accessing the model using gateway

Gateway is designed to serve LLM requests and provides features such as dynamic model & LoRA adapter discovery, user configuration for request count & token usage budgeting, streaming and advanced routing strategies such as prefix-cache aware, heterogeneous GPU hardware.
To access the model using Gateway, Please run the below command

```bash
kubectl -n envoy-gateway-system port-forward service/envoy-aibrix-system-aibrix-eg-903790dc 8888:80 &
```

Once the port-forward is running, you can test the model by sending a request to the Gateway.

```bash
ENDPOINT="localhost:8888"
curl -v http://${ENDPOINT}/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek-r1-distill-llama-8b",
        "prompt": "San Francisco is a",
        "max_tokens": 128,
        "temperature": 0
    }'
```


<CollapsibleContent header={<h2><span>Cleanup</span></h2>}>

This script will cleanup the environment using `-target` option to ensure all the resources are deleted in correct order.

```bash
kubectl delete -f blueprints/inference/aibrix/deepseek-distill.yaml
```

To cleanup the AIBrix deployment, and delete the EKs cluster please run the below command

```bash
cd infra/aibrix/terraform
./cleanup.sh
```

</CollapsibleContent>

:::caution
To avoid unwanted charges to your AWS account, delete all the AWS resources created during this deployment
:::
