### Secure Accelerator Access - Test 2

Create two Pods, each is allocated an accelerator resource. Execute a command in one Pod to attempt to access the other Pod’s
accelerator, and should be denied.

**Step 1**: Make Kubernetes DRA e2e tests

```
git clone https://github.com/kubernetes/kubernetes
```
```
git checkout v1.34.1
```
```
make WHAT="ginkgo k8s.io/kubernetes/test/e2e/e2e.test"
```

**Step 2**: Run multi-container access test

```
KUBECONFIG=/Users/xxx/.kube/config _output/bin/ginkgo -v -focus='must map configs and devices to the right containers' ./test/e2e
```

```
...
  I1010 16:34:11.524263    9569 e2e.go:109] Starting e2e run "ee4a01d2-cde8-4fe6-8c93-b30814c0ea9d" on Ginkgo node 1
Running Suite: Kubernetes e2e suite - /Users/xxx/kubernetes/test/e2econtainers'
  I1010 16:34:12.969060 9569 e2e.go:142] Waiting up to 5m0s for all daemonsets in namespace 'kube-system' to start
  I1010 16:34:13.018830 9569 e2e.go:153] 1 / 1 pods ready in namespace 'kube-system' in daemonset 'aws-node' (0 seconds elapsed)
  I1010 16:34:13.018897 9569 e2e.go:153] 1 / 1 pods ready in namespace 'kube-system' in daemonset 'kube-proxy' (0 seconds elapsed)
  I1010 16:34:13.018914 9569 e2e.go:245] e2e test version: v0.0.0-master+$Format:%H$
  I1010 16:34:13.062742 9569 e2e.go:254] kube-apiserver version: v1.34.1-eks-d96d92f
  [sig-node] [DRA] kubelet [Feature:DynamicResourceAllocation] must map configs and devices to the right containers [sig-node, DRA, Feature:DynamicResourceAllocation]
/Users/xxx/kubernetes/test/e2e/dra/dra.go:331
  STEP: Creating a kubernetes client @ 10/10/25 16:34:13.176
  I1010 16:34:13.176778 9569 util.go:414] >>> kubeConfig: /Users/xxx/.kube/config
  STEP: Building a namespace api object, basename dra @ 10/10/25 16:34:13.177
  STEP: Waiting for a default service account to be provisioned in namespace @ 10/10/25 16:34:13.323
  STEP: Waiting for kube-root-ca.crt to be provisioned in namespace @ 10/10/25 16:34:13.411
  STEP: selecting nodes @ 10/10/25 16:34:13.502
  I1010 16:34:13.550318 9569 deploy.go:142] testing on nodes [ip-192-168-38-195.us-west-2.compute.internal]
  STEP: deploying driver dra-3728.k8s.io on nodes [ip-192-168-38-195.us-west-2.compute.internal] @ 10/10/25 16:34:13.55
  I1010 16:34:13.596944    9569 deploy.go:154] "Listed ResourceClaims" logger="ResourceClaimListWatch" resourceAPI="V1" numClaims=0 listMeta={"resourceVersion":"14302"}
  I1010 16:34:13.642150    9569 deploy.go:163] "Started watching ResourceClaims" logger="ResourceClaimListWatch" resourceAPI="V1"
  I1010 16:34:13.953068 9569 create.go:156] creating *v1.ReplicaSet: dra-3728/dra-test-driver
  STEP: wait for plugin registration @ 10/10/25 16:34:16.173
  STEP: creating *v1.DeviceClass dra-3728-class @ 10/10/25 16:34:18.174
  STEP: creating *v1.ResourceClaim all @ 10/10/25 16:34:18.224
  STEP: creating *v1.ResourceClaim container0 @ 10/10/25 16:34:18.275
  STEP: creating *v1.ResourceClaim container1 @ 10/10/25 16:34:18.324
  STEP: creating *v1.Pod tester-1 @ 10/10/25 16:34:18.375
  STEP: delete pods and claims @ 10/10/25 16:34:27.543
  STEP: deleting *v1.Pod dra-3728/tester-1 @ 10/10/25 16:34:27.594
  STEP: deleting *v1.ResourceClaim dra-3728/all @ 10/10/25 16:34:31.85
  STEP: deleting *v1.ResourceClaim dra-3728/container0 @ 10/10/25 16:34:31.903
  STEP: deleting *v1.ResourceClaim dra-3728/container1 @ 10/10/25 16:34:31.956
  STEP: waiting for resources on ip-192-168-38-195.us-west-2.compute.internal to be unprepared @ 10/10/25 16:34:32.011
  STEP: waiting for claims to be deallocated and deleted @ 10/10/25 16:34:32.012
  STEP: scaling down driver proxy pods for dra-3728.k8s.io @ 10/10/25 16:34:32.646
  STEP: Waiting for ResourceSlices of driver dra-3728.k8s.io to be removed... @ 10/10/25 16:34:33.175
  STEP: Destroying namespace "dra-3728" for this suite. @ 10/10/25 16:34:33.318
...

Ran 1 of 7193 Specs in 21.824 seconds
SUCCESS! -- 1 Passed | 0 Failed | 0 Pending | 7192 Skipped
PASS

Ginkgo ran 1 suite in 32.620402417s
Test Suite Passed
```
