---
sidebar_label: Building and Deploying Agents
---

# Building and Deploying Agents on EKS

This guide walks through how to use the Agents on EKS environment to take an agent you've developed and get it running
in production. It covers the end-to-end workflow: pushing code to source control, building container images via CI/CD,
deploying to Kubernetes, and configuring access to AWS services.

## Prerequisites

Before starting, you should have:

- The [Agents on EKS infrastructure](https://awslabs.github.io/ai-on-eks/docs/infra/agents-on-eks) deployed
- An agent that runs locally (see [Best Practices for Agent Development](./best-practices.md) for guidance on
  structuring your agent code with the [Strands Agents SDK](https://strandsagents.com))
- `kubectl` configured to access your EKS cluster
- Docker Desktop or Podman installed locally

:::note
We've tried to minimize the surface of the environment that is exposed to the public internet; therefore, we will be
using port-forwarding as much as possible. This is also why the instructions have you set your IP address as an inbound
CIDR range allowed by the load balancer. It is possible to remove IP restrictions or expose more of the environment to
the public internet (LangFuse or your agents, for instance). You will need to create ingresses yourself if you want to
do that.
:::

## Containerize Your Agent

Before your agent can run in the environment, it needs to be packaged as a container image. If you've been maintaining a
`requirements.txt` for your dependencies, you just need a `Dockerfile`:

```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt /app
RUN pip install -r requirements.txt
COPY * /app
CMD python entrypoint.py
```

:::tip
Check the output of `python -V` locally and match the version in the `FROM` line to avoid compatibility issues between
your local environment and the container.
:::

Build and test locally:

```bash
docker build -t strands-agent .
docker run --rm -p 8000:8000 \
  -e AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY \
  -e AWS_REGION \
  strands-agent
```

Verify the agent responds correctly before moving on. Once it works in a container locally, it will work the same way in
the environment.

## Push Code to GitLab

The environment includes a GitLab instance for source control, container registry, and CI/CD. Log in to GitLab and
create a new project:

To get the `root` user's password:

```bash
kubectl get secret -n gitlab gitlab-gitlab-initial-root-password -o jsonpath="{.data.password}" | base64 --decode
```

Log in as username: root, password from above

![GitLab new project step 1](img/gitlab-new-project-1.png)

![GitLab new project step 2](img/gitlab-new-project-2.png)

![GitLab new project step 3](img/gitlab-new-project-3.png)

![GitLab new project step 4](img/gitlab-new-project-4.png)

![GitLab new project step 5](img/gitlab-new-project-5.png)

![GitLab new project step 6](img/gitlab-new-project-6.png)

Clone the repository and add your agent files:

```bash
git clone https://gitlab.<your-domain>/root/strands-agent.git
cd strands-agent

# Copy your agent files into this directory, then:
git add .
git commit -m "initial commit"
git push origin main
```

## Set Up CI/CD

Add a `.gitlab-ci.yml` file to your repository so GitLab automatically builds a container image and pushes it to the
internal registry on every commit:

```yaml
build-rootless:
  image: moby/buildkit:rootless
  stage: build
  variables:
    BUILDKITD_FLAGS: --oci-worker-no-process-sandbox
  before_script:
    - mkdir -p ~/.docker
    - echo "{\"auths\":{\"$CI_REGISTRY\":{\"username\":\"$CI_REGISTRY_USER\",\"password\":\"$CI_REGISTRY_PASSWORD\"}}}" > ~/.docker/config.json
  script:
    - |
      buildctl-daemonless.sh build \
        --frontend dockerfile.v0 \
        --local context=. \
        --local dockerfile=. \
        --output type=image,name=$CI_REGISTRY_IMAGE:$CI_COMMIT_SHA,push=true
```

Push the pipeline file:

```bash
git add .gitlab-ci.yml
git commit -m "add CI/CD pipeline"
git push
```

After the next push, navigate to the pipeline view in GitLab to see the build run:

![GitLab pipeline view](img/gitlab-pipeline.png)

Click the green checkmark on your pipeline stage and click on the job name to see the build output. At the bottom you'll
find the full image name in the registry:

```text
pushing manifest for registry.domain.tld/root/strands-agent:<commit-sha>@sha256:<digest>
```

You can verify the image works by pulling and running it locally. The first time you `docker pull`, you'll be prompted
for your GitLab credentials:

```bash
docker run --rm -p 8000:8000 \
  -e AWS_ACCESS_KEY_ID \
  -e AWS_SECRET_ACCESS_KEY \
  -e AWS_REGION \
  registry.domain.tld/root/strands-agent:<commit-sha>
```

From this point on, every commit to your repository will produce a new tagged image in the registry, ready to deploy.

## Deploy to Kubernetes

### Create a Namespace

Namespaces group resources together and provide isolation. Create one for your agent:

```bash
kubectl create namespace strands-agent
```

### Create a Service Account

The service account handles both pulling images from the registry and authenticating to AWS services via Pod Identity:

```bash
kubectl create serviceaccount -n strands-agent strands-agent
```

### Configure Registry Access

Create a [deploy token](https://docs.gitlab.com/user/project/deploy_tokens/#create-a-deploy-token) in GitLab with
`read_registry` scope. Then create a Kubernetes secret with those credentials:

```bash
kubectl create secret docker-registry regcred \
  -n strands-agent \
  --docker-server=registry.<your-domain> \
  --docker-username=gitlab+deploy-token-1 \
  --docker-password=<TOKEN>
```

Associate the credentials with the service account:

```bash
kubectl patch serviceaccount -n strands-agent strands-agent \
  -p '{"imagePullSecrets": [{"name": "regcred"}]}'
```

:::note
Each GitLab repository requires its own deploy token. This is the most secure approach, scoping registry access to a
single repository, but does require a token per repo as you scale.
:::

### Create the Deployment

Save the following as `deployment.yaml`, replacing the image reference with your registry image:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: strands-agent
  namespace: strands-agent
  labels:
    app: strands-agent
spec:
  replicas: 1
  selector:
    matchLabels:
      app: strands-agent
  template:
    metadata:
      labels:
        app: strands-agent
    spec:
      containers:
        - name: agent
          image: registry.<your-domain>/root/strands-agent:<commit-sha>
          ports:
            - containerPort: 8000
      serviceAccountName: strands-agent
---
apiVersion: v1
kind: Service
metadata:
  name: strands-agent
  namespace: strands-agent
spec:
  ports:
    - port: 8000
      targetPort: 8000
      protocol: TCP
      name: http
  selector:
    app: strands-agent
```

Deploy it:

```bash
kubectl apply -f deployment.yaml
```

## Configure AWS Access with Pod Identity

If your agent needs access to AWS services (Bedrock, S3, etc.),
use [Pod Identity](https://docs.aws.amazon.com/eks/latest/userguide/pod-identities.html) rather than passing IAM
credentials as environment variables. The Pod Identity agent is already running in the environment.

When creating the Pod Identity Role, make sure to add Bedrock permissions to the role.

Redeploy the agent to pick up the new credentials:

```bash
kubectl rollout restart deployment/strands-agent -n strands-agent
```

## Verify the Deployment

Check that the pod is running:

```bash
kubectl get pods -n strands-agent
```

```text
NAME                        READY   STATUS    RESTARTS   AGE
strands-agent-7ddfd847b4-9x4xs   1/1     Running   0          2m36s
```

Check the logs:

```bash
kubectl logs -n strands-agent -l app=strands-agent
```

```text
INFO:     Started server process [7]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Test it via port-forward:

```bash
kubectl port-forward -n strands-agent svc/strands-agent 8000
```

In another terminal:

```bash
curl --location 'http://localhost:8000/' \
--header 'Content-Type: application/json' \
--data '{"request": "Hello"}'
```

Your agent is now running in Kubernetes and accessible to any service in the cluster.

## Updating Your Agent

The workflow for updating a deployed agent is:

1. Make code changes locally and test them
2. Commit and push to GitLab
3. The CI/CD pipeline builds a new image tagged with the commit SHA
4. Update the image reference in `deployment.yaml` and re-apply, or use `kubectl set image` to update in place:

```bash
kubectl set image deployment/strands-agent \
  -n strands-agent \
  agent=registry.<your-domain>/root/strands-agent:<new-commit-sha>
```

## What's Next?

- [Best Practices for Agent Development](./best-practices.md) — Patterns for structuring agent code, testing, and API
  wrapping
