# Observability

Observability for AI/ML workloads requires a holistic view of multiple hardware/software components alongside multiple sources of data such as logs, metrics, and traces. Piecing together these components is challenging and time-consuming; therefore, we leverage the [AI/ML Observability](https://github.com/awslabs/ai-ml-observability-reference-architecture) available in Github to bootstrap this environment.

## Architecture
![Architecture](https://github.com/awslabs/ai-ml-observability-reference-architecture/raw/main/static/reference_architecture.png)

## What's Included
- Prometheus
- OpenSearch
- FluentBit
- Kube State Metrics
- Metrics Server
- Alertmanager
- Grafana
- Pod/Service monitors for AI/ML workloads
- AI/ML Dashboards

## Why
Understanding the performance of AI/ML workloads is challenging: Is the GPU getting data fast enough? Is the CPU the bottleneck? Is the storage fast enough? These are questions that are hard to answer in isolation. The more of the picture one is able to see, the more clarity there is in identifying performance bottlenecks.

## How
The [JARK](https://awslabs.github.io/ai-on-eks/docs/infra/jark) infrastructure already comes with this architecture enabled by default, if you would like to add it to your infrastructure, you need to ensure 2 variables are set to `true` in `blueprint.tfvars`:

```yaml
enable_argocd                    = true
enable_ai_ml_observability_stack = true
```

The first variable deploys ArgoCD, which is used to deploy the observability architecture, the second variable deploys the architecture.

## Usage
The architecture is entirely deployed into the `monitoring` namespace. To access Grafana: `kubectl port-forward -n monitoring svc/kube-prometheus-stack-grafana 3000:80`. You can then open https://localhost:3000 to log into grafana with username `admin` and password `prom-operator`. You can refer to the [security](https://github.com/awslabs/ai-ml-observability-reference-architecture?tab=readme-ov-file#security) section in the Readme to see how to change the username/password

### Training
Ray training job logs and metrics will be automatically collected by the Observability architecture and can be found in the [training dashboard](http://localhost:3000/d/ee6mbjghme96oc/gpu-training?orgId=1&refresh=5s&var-namespace=default&var-job=ray-train&var-instance=All).

#### Example
A full example of this can be found in the [AI/ML observability repo](https://github.com/awslabs/ai-ml-observability-reference-architecture/tree/main/examples/training). We will also be updating the Blueprints here to make use of this architecture.

### Inference
Ray inference metrics should be automatically picked up by the observability infrastructure and can be found in the [inference dashboard](http://localhost:3000/d/bec31e71-3ac5-4133-b2e3-b9f75c8ab56c/inference-dashboard?orgId=1&refresh=5s). To instrument your inference workloads for logging, you will need to add a few items:

#### FluentBit Config
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentbit-config
  namespace: default
data:
  fluent-bit.conf: |-
    [INPUT]
        Name tail
        Path /tmp/ray/session_latest/logs/*
        Tag ray
        Path_Key true
        Refresh_Interval 5
    [FILTER]
        Name modify
        Match ray
        Add POD_LABELS ${POD_LABELS}
    [OUTPUT]
        Name stdout
        Format json
```
Deploy this into the namespace in which you intend to run your inference workload. You only need one in each namespace to tell the FluentBit sidecar how to output the logs.

#### FluentBit Sidecar
We will need to add a sidecar to the Ray inference service so FluentBit can write the logs to STDOUT
```yaml
              - name: fluentbit
                image: fluent/fluent-bit:3.2.2
                env:
                  - name: POD_LABELS
                    valueFrom:
                      fieldRef:
                        fieldPath: metadata.labels['ray.io/cluster']
                resources:
                  requests:
                    cpu: 100m
                    memory: 128Mi
                  limits:
                    cpu: 100m
                    memory: 128Mi
                volumeMounts:
                  - mountPath: /tmp/ray
                    name: ray-logs
                  - mountPath: /fluent-bit/etc/fluent-bit.conf
                    subPath: fluent-bit.conf
                    name: fluentbit-config
```
Add this section to the `workerGroupSpecs` containers

#### FluentBit Volume
Finally, we need to add the configmap volume to our `volumes` section:
```yaml
              - name: fluentbit-config
                configMap:
                  name: fluentbit-config
```

#### vLLM Metrics
vLLM also outputs useful metrics like the Time to First Token, throughput, latencies, cache utilization and more. To get access to these metrics, we need to add a route to our pod for the metrics path:

```python
# Imports
import re
from prometheus_client import make_asgi_app
from fastapi import FastAPI
from starlette.routing import Mount

app = FastAPI()

class Deployment:
    def _init__(selfself, **kwargs):
        ...
        route = Mount("/metrics", make_asgi_app())
        # Workaround for 307 Redirect for /metrics
        route.path_regex = re.compile('^/metrics(?P<path>.*)$')
        app.routes.append(route)
```

This will allow the deployed monitor to collect the vLLM metrics and display them in the inference dashboard.

#### Example
A full example of this can be found in the [AI/ML observability repo](https://github.com/awslabs/ai-ml-observability-reference-architecture/tree/main/examples/inference). We will also be updating the Blueprints here to make use of this architecture.
