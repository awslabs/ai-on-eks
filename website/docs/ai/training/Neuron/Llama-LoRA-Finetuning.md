---
sidebar_position: 1
sidebar_label: Llama 3 Fine-tuning with LoRA
---
import CollapsibleContent from '../../../../src/components/CollapsibleContent';

:::warning
Deployment of ML models on EKS requires access to GPUs or Neuron instances. If your deployment isn't working, it’s often due to missing access to these resources. Also, some deployment patterns rely on Karpenter autoscaling and static node groups; if nodes aren't initializing, check the logs for Karpenter or Node groups to resolve the issue.
:::

:::danger

Note: Use of this Llama 3 model is governed by the Meta license.
In order to download the model weights and tokenizer, please visit the [website](https://ai.meta.com/) and accept the license before requesting access.

:::

:::info

We are actively enhancing this blueprint to incorporate improvements in observability, logging, and scalability aspects.

:::

# Llama 3 fine-tuning on Trn1 with HuggingFace Optimum Neuron

This comprehensive guide walks you through the steps for fine-tuning the `Llama3-8B` language model using AWS Trainium (Trn1) EC2 instances. The fine-tuning process is facilitated by HuggingFace Optimum Neuron, a powerful library that simplifies the integration of Neuron into your training pipeline.

### What is Llama 3?

Llama 3 is a state-of-the-art large language model (LLM) designed for various natural language processing (NLP) tasks, including text generation, summarization, translation, question answering, and more. It's a powerful tool that can be fine-tuned for specific use cases.

#### AWS Trainium:
- **Optimized for Deep Learning**: AWS Trainium-based Trn1 instances are specifically designed for deep learning workloads. They offer high throughput and low latency, making them ideal for training large-scale models like Llama 3. Trainium chips provide significant performance improvements over traditional processors, accelerating training times.
- **Neuron SDK**: The AWS Neuron SDK is tailored to optimize your deep learning models for Trainium. It includes features like advanced compiler optimizations and support for mixed precision training, which can further accelerate your training workloads while maintaining accuracy.

## 1. Deploying the Solution

<CollapsibleContent header={<h2><span>Prerequisites</span></h2>}>
Before we begin, you will need to ensure you have all the prerequisites in place to make the deployment process smooth and hassle-free. You will need a machine from where you will be driving this solution deployment and interacting with the container that will run the Llama 3 fine-tuning code. You can use a [EC2 Instance](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html), a local Mac machine, or Windows machine. Ensure that you have Docker installed locally with storage above 100GB and that the image is created with x86 architecture. We'll assume that it is a EC2 instance for the rest of this exercise.

Ensure that you have installed the following tools on this EC2 instance:

* [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
* [kubectl](https://Kubernetes.io/docs/tasks/tools/)
* [terraform](https://learn.hashicorp.com/tutorials/terraform/install-cli)
* Git(Only for EC2 instance)
* Docker
* Python, pip, jq, unzip

To install all the pre-reqs on EC2, you can run this [script](https://github.com/VijoyChoyi/ai-on-eks/blob/main/infra/trainium-inferentia/examples/llama2/install-pre-requsites-for-ec2.sh) which is compatible with Amazon Linux 2023.


**Clone the Data on EKS repository**

After tackling the pre-requisites, we can get started by cloning the ai-on-eks github repository.

```bash
git clone https://github.com/awslabs/ai-on-eks.git
```

**Navigate to the trainium-inferentia directory.**

```bash
cd ai-on-eks/infra/trainium-inferentia
```

Set the region value within the `blueprint.tfvars` variables file under the `terraform` sub-folder to match your preference.

**NOTE:** Trainium instances are available in select regions, and the user can determine this list of regions using the commands outlined [here](https://repost.aws/articles/ARmXIF-XS3RO27p0Pd1dVZXQ/what-regions-have-aws-inferentia-and-trainium-instances) on re:Post.


Run the installation script to provision an EKS cluster with all the add-ons needed for the solution.

```bash
./install.sh
```

### Verify the resources

Verify the Amazon EKS Cluster

```bash
aws eks --region us-west-2 describe-cluster --name trainium-inferentia
```

```bash
# Creates k8s config file to authenticate with EKS
aws eks --region us-west-2 update-kubeconfig --name trainium-inferentia

kubectl get nodes # Output shows the EKS Managed Node group nodes
```

</CollapsibleContent>

## 2. Build the Docker Image

We'll build the docker image that will be used by the container to run Llama 3 fine-tuning. Execute the below commands after ensuring you are in the root folder of the ai-on-eks repository.

**NOTE:** Before running the docker image builder script, make sure the AWS user or principal used to run the script has proper access to the ECR repository in the specific region you selected earlier. The script will create a repo and push the image into it.

```bash
cd blueprints/training/llama-lora-finetuning-trn1
./build-container-image.sh
```
After running this script, note the Docker image URL and tag that gets displayed at the end of the run. You will need this information for the next step.

## 3. Launch the Llama training pod

Update the container image value in the `lora-finetune-pod.yaml` file with the Docker image URL and tag obtained from the previous step.

Utilize kubectl cli to launch the `lora-finetune-app` in your EKS cluster:

```bash
kubectl apply -f lora-finetune-pod.yaml
```

## 4. Launch LoRA fine-tuning

**Verify the Pod Status:**

```bash
kubectl get pods

```

**NOTE:** If the pod fails to get scheduled, check the Karpenter logs for errors. Besides hitting resource limits or other common reasons, it's also possible that the infrastructure that was setup in the initial step through terraform ended up choosing a subnet whose AZ doesn't have trn1.32xlarge compute capacity. In such cases, you will need to update the `trainium-trn1` EC2NodeClass's subnet to one that is associated with an AZ that has capacity. You can find this EC2NodeClass definition in the `addons.tf` that gets copied under `infra/trainium-inferentia/terraform` folder. You'll need to re-run the `install.sh` script in that folder after making this update and then see if the pod gets scheduled in the new AZ/subnet by Karpenter.

Once the pod is in ‘Running’ state, connect to it using an interactive bash command shell:

```bash
kubectl exec -it lora-finetune-app -- /bin/bash
```

Before launching the fine-tuning script `01__launch_training.sh`, you need to set an environment variable with your HuggingFace token. The access token is found under Settings → Access Tokens on the Hugging Face website after you login to the HuggingFace website.

```bash
export HF_TOKEN=<your-huggingface-token>

./01__launch_training.sh
```

Once the script is complete, you can verify the training progress by checking the logs of the training job.

Next, we need to consolidate the adapter shards and merge the model. For this we run the python script `02__consolidate_adapter_shards_and_merge_model.py` by passing in the location of the checkpoint using the '-i' parameter and providing the location where you want to save the consolidated model using the '-o' parameter.
```
python3 ./02__consolidate_adapter_shards_and_merge_model.py -i /neuron/finetuned_models/20250220_170215/checkpoint-250/ -o /neuron/finetuned_models/tuned_model
```

Once the script is complete, we can test the fine-tuned model by running the `03__test_model.py` by passing in the location of the tuned model using the '--tuned-model' parameter.
```bash
python3 ./03__test_model.py --tuned-model /neuron/finetuned_models/tuned_model
```

You can exit from the interactive terminal of the pod once you are done testing the model.

### Cleaning up

To remove the resources created using this solution, execute the below commands after ensuring you are in the root folder of the ai-on-eks repository.

```bash
# Delete the Kubernetes Resources:
cd blueprints/training/llama-lora-finetuning-trn1
kubectl delete -f lora-finetune-pod.yaml

# Clean Up the EKS Cluster and Associated Resources:
cd ../../../infra/base/terraform
./cleanup.sh
```
