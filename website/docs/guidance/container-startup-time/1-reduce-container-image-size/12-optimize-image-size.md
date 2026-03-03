---
sidebar_label: Optimizing container images size
---

# Optimizing container images size

## Selecting appropriate base images

Different AI/ML frameworks and platforms offer ready-to-use container images, providing convenience and enabling experimentation. However, these images often try to address as wide a feature set as possible and thus may include different runtimes, frameworks, or supported APIs, leading to bloat.

For example, different PyTorch image variants have very different sizes: from [2.7.1-cuda11.8-cudnn9-devel](https://hub.docker.com/layers/pytorch/pytorch/2.7.1-cuda11.8-cudnn9-devel/images/sha256-5a046e4e3364b063a17854387b8820ad3f42ed197a089196bce8f2bd68f275a8) (6.66 GB) that includes development tools, compilers etc. to [2.7.1-cuda11.8-cudnn9-runtime](https://hub.docker.com/layers/pytorch/pytorch/2.7.1-cuda11.8-cudnn9-runtime/images/sha256-8d409f72f99e5968b5c4c9396a21f4b723982cfdf2c1a5b9cc045c5d0a7345a1) (3.03 GB) that contains runtime only. The vLLM project [provides](https://docs.vllm.ai/en/stable/contributing/dockerfile/dockerfile.html) several container image variants, each with different capabilities packaged in, such as the OpenAI spec, Sagemaker integration, and more.

Selecting a smaller base image that satisfies the needs of the application can make a big difference, with a caveat – smaller runtime-only images may not include JIT compilation or dynamic optimization and thus fall on the slower code paths, reducing startup time.

A comprehensive approach would include:

* benchmarking workloads with different base images
* considering custom builds that include only required optimization libraries
* testing overall cold start performance in addition to image pull time improvements

## Using multi-stage builds

Multi-stage container image builds, supported by multiple platforms such as Docker/BuildKit, Podman, Finch/Buildkit, allow use of multiple FROM statements in a single container image file to separate build process and artifacts from the runtime concerns.

A multi-stage build container image file may look similar to the following:

```
# Build stage that also uses an external image to copy artifacts
FROM python:3.12-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.7.11 /uv /uvx /bin/
...

# Runtime stage
FROM python:3.12-slim-bookworm
...
COPY --from=models some-model /app/models/some-model/configs
COPY --from=builder --chown=app:app /app/.venv ./.venv
COPY --from=builder --chown=app:app /app/main.py ./main.py
...
CMD ["sh", "-c", "exec fastapi run --host 0.0.0.0 --port 80 /app/main.py"]
```

:::info
Unlike typical application dependencies baked into container images, copying large model files (ranging from a few GBs to tens of GBs) is generally discouraged. This is due to increased container image size which affect its pull time, separate release lifecycle for the app and model, and potential storage duplication when sharing models across multiple apps.
:::

Copying only the required artifacts allows fine-grained control over which components of the build result are to be included in the final runtime image, reducing its size (along with other benefits, like security or workflow simplicity).

In the example above we also employed two other [variations](https://docs.docker.com/reference/dockerfile/#copy---from) of `COPY --from`  (supported via BuildKit by the majority of popular image building platforms) :

* `COPY --from=<path/to/image/in/registry and parts to be taken>` allows to extract only the specific files and folders from another container image stored in a registry
* `COPY --from=<name of the build context>` allows to copy only the specific files and folders from a local folder, provided as a parameter to the build command using `--build-context models=/path/to/local/folder`

Note that while using `.dockerignore` is a good practice, in general, and should be used alongside the above process, it doesn’t impact the `COPY --from=...` commands.

Additionally this technique can be further (if sometimes marginally) improved by using the following one, taking caveats into consideration.

## Employing layer optimization techniques

As image layers (now smaller in total size) are downloaded during the pull process, they are decompressed and unpacked to assemble the container’s file system. The amount and size of the image layers have an impact of the duration of that process, serving as another candidate for optimization.

One commonly mentioned optimization involves combining the `RUN` or `COPY` commands to create smaller amount of larger layers, with the following typical example:

```
FROM ...
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libssl-dev \
        pkg-config && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean
```

In context of the recommendation to employ multi-stage builds, `RUN` isn’t often a part of the last – runtime stage. This is because, ideally, all execution is done during the previous build stages and the processed artifacts are only copied into their intended locations in the runtime stage.

The `COPY` command, in the runtime stage, can be optimized using this process, with one challenge – it doesn’t support multiple destinations or multiple source stages, so combining these commands into one isn’t possible:

```
COPY --from=models some-model /app/models/some-model/configs
COPY --from=builder1 --chown=app:app /app/.venv ./.venv
COPY --from=builder1 --chown=app:app /app/main.py ./main.py
COPY --from=builder2 --chown=app:app /app/config.json ./config.json
```

To overcome this, an additional copying stage may be introduced, where the final folder structure is created and then copied via a single command:

```
FROM python:3.12-bookworm AS builder1
WORKDIR /app
...

FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel AS builder2
WORKDIR /app
...

FROM scratch AS assembly
...
COPY --from=models some-model/weights /app/models/some-model/weights
COPY --from=builder1 /app/.venv /app/venv
COPY --from=builder1 /app/main.py /app/main.py
COPY --from=builder2 /dist/config.json /app/config.json

FROM python:3.12-slim-bookworm
COPY --from=assembly --chown=app:app /app /app
CMD ["python", "main.py"]
```

Take into consideration that even when the above steps have a positive effect on the overall container startup time, it is often negligible, relative to other solutions in this guide and should be assessed before investing time into the technique.

The improvement should be weighted against the trade-offs that include:

* reduction in cache efficiency, due to its lower granularity with a smaller amount of larger layers if layers are not ordered correctly
* build time, due to more shuffling for the sake of optimization of the runtime layer
