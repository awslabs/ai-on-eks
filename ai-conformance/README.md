# Kubernetes AI Conformance evidence

Evidence supporting the Amazon EKS submissions to the CNCF [Certified Kubernetes AI Platform](https://github.com/cncf/k8s-ai-conformance) program. Each submission is a `PRODUCT.yaml` in `cncf/k8s-ai-conformance` under `vX.Y/eks/`; the files here are the material its `evidence` and `notes` fields point at.

## Layout

```
ai-conformance/
  <version>/<requirement-or-activity>/
    Readme.md        what was run, on which cluster, how to reproduce it
    *.yaml, *.sh     the manifests and scripts that produced the result
    e2e.log          test execution log          (automated runs)
    junit.xml        JUnit report                (automated runs)
    results.json     go test -json output        (automated runs)
```

| Version | Directory | How it was verified |
|---|---|---|
| 1.34 | `secure_accelerator_access`, `pod_autoscaling` | Procedures walked by hand, with recorded output |
| 1.36 | `automated-tests` | Upstream [kubernetes-sigs/ai-conformance](https://github.com/kubernetes-sigs/ai-conformance) suite |
| 1.37 | `automated-tests` | Upstream suite |

From Kubernetes 1.37 the program supports a hybrid approach: requirements with an automated test are verified by running the upstream suite and attaching its artifacts, and the rest by public documentation. The 1.36 and 1.37 directories therefore hold suite output, where 1.34 holds hand-walked procedures.

## Reading a result

Each directory's `Readme.md` names the cluster, the Kubernetes and EKS platform version, the accelerator node and the allocation mechanism, because those determine what the run proves. Where a requirement asks whether the platform can run a given solution, the component installed to exercise it is named too — these are not components EKS ships.

Artifacts are the output of the upstream suite, unedited apart from whitespace normalization for this repository's pre-commit hooks.
