# old doc: https://pipekit.io/blog/using-helm-charts-to-deploy-argo-workflows
# https://github.com/argoproj/argo-helm
# https://argo-workflows.readthedocs.io/en/latest/installation/
# https://argo-workflows.readthedocs.io/en/latest/walk-through/volumes/

helm repo add argo https://argoproj.github.io/argo-helm

helm search repo argo
# argo/argo                       1.0.0           v2.12.5         A Helm chart for Argo Workflows
# argo/argo-cd                    6.6.0           v2.10.2         A Helm chart for Argo CD, a declarative, GitOps...
# argo/argo-ci                    1.0.0           v1.0.0-alpha2   A Helm chart for Argo-CI
# argo/argo-events                2.4.3           v1.9.1          A Helm chart for Argo Events, the event-driven ...
# argo/argo-lite                  0.1.0                           Lighweight workflow engine for Kubernetes
# argo/argo-rollouts              2.34.3          v1.6.6          A Helm chart for Argo Rollouts
# argo/argo-workflows             0.40.14         v3.5.5          A Helm chart for Argo Workflows
# argo/argocd-applicationset      1.12.1          v0.4.1          A Helm chart for installing ArgoCD ApplicationSet
# argo/argocd-apps                1.6.2                           A Helm chart for managing additional Argo CD Ap...
# argo/argocd-image-updater       0.9.5           v0.12.2         A Helm chart for Argo CD Image Updater, a tool ...
# argo/argocd-notifications       1.8.1           v1.2.1          A Helm chart for ArgoCD notifications, an add-o...

helm -n argo install \
   --namespace argo \
   --create-namespace \
   argo-worklows argo/argo-workflows


kubectl --namespace argo get services -o wide | grep argo-worklows-argo-workflows-server

# create default service account
kubectl -n argo create rolebinding default-admin --clusterrole=admin --serviceaccount=argo:default 

# patch svc as load-balancer
kubectl -n argo patch svc argo-worklows-argo-workflows-server  -p '{"spec": {"type": "LoadBalancer"}}'

# change auth mode of server
# https://argo-workflows.readthedocs.io/en/latest/argo-server-auth-mode/
# https://argoproj.github.io/argo-workflows/argo-server-auth-mode/
# https://github.com/argoproj/argo-workflows/issues/9868
cat <<EOF | kubectl create -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: workflow-controller-configmap
  namespace: argo
data: {}
EOF

kubectl -n argo patch deployment argo-server \
   --namespace argo \
   --type='json' \
   -p='[{"op": "replace", "path": "/spec/template/spec/containers/0/args", "value": [
   "server",
   "--auth-mode=server"
   ]}]'

export ARGO_SECURE=true
argo -n argo server --auth-mode=server

# create role, service account, and role-binding
# copy from install role by argo and edit
# add create support to persistent volume claim
# kubectl -n argo get role argo-worklows-argo-workflows-workflow -o yaml > argo-role.yaml
kubectl -n argo apply -f argo-role.yaml

# create service account
kubectl -n argo create sa argo-sa
# kubectl -n argo create sa argo-workflow

# create role-binding
cat > role-binding.yaml <<HERE
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: argo-role-binding
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: argo-role
subjects:
- kind: ServiceAccount
  name: argo-sa
HERE

kubectl -n argo apply -f role-binding.yaml

export ARGO_SECURE=true
export ARGO_INSECURE_SKIP_VERIFY=true
argo -n argo --secure --insecure-skip-verify list 

export ARGO_SECURE=true
argo -n argo server --auth-mode=server

# https://docs.coreweave.com/cloud-tools/argo/security-best-practices-for-argo-workflows
# argo -n argo submit --serviceaccount argo-workflow https://raw.githubusercontent.com/argoproj/argo-workflows/master/examples/hello-world.yaml --watch
# argo -n argo submit --serviceaccount argo-sa https://raw.githubusercontent.com/argoproj/argo-workflows/master/examples/hello-world.yaml --watch

# Use Volumes instead of Minio
# https://argo-workflows.readthedocs.io/en/latest/walk-through/volumes/
# argo -n argo submit --serviceaccount argo-sa ./argo-workflow-example-volumeclaim1.yaml --watch

# delete all workflows
argo -n argo delete --all

# examples
argo -n argo submit ./yaml-examples/hello1.yaml --watch
argo -n argo submit --serviceaccount default ./yaml-examples/hello1.yaml --watch
argo -n argo submit ./yaml-examples/hello1.yaml -p message="hello universe" --watch --log
argo -n argo submit ./yaml-examples/hello2.yaml  --watch
argo -n argo submit ./yaml-examples/dag-parallel.yaml  --watch
argo -n argo submit ./yaml-examples/run-scripts.yaml  --watch
argo -n argo submit ./yaml-examples/output-parameters.yaml  --watch
# secret
kubectl -n argo create secret generic my-secret --from-literal=mypassword=S00perS3cretPa55word
argo -n argo submit ./yaml-examples/secret-example.yaml  --watch
argo -n argo submit --serviceaccount argo-sa ./yaml-examples/argo-workflow-example-volumeclaim1.yaml --watch
# create pvc first called my-existing-volume
kubectl -n argo apply -f ./yaml-examples/pvc.yaml
argo -n argo submit --serviceaccount argo-sa ./yaml-examples/argo-workflow-example-volumeclaim2.yaml --watch
kubectl -n argo delete -f ./yaml-examples/pvc.yaml
# template level
argo -n argo submit --serviceaccount argo-sa ./yaml-examples/argo-workflow-example-volumeclaim3.yaml --watch
argo -n argo logs  template-level-volume-dh269
# sequence example
argo -n argo submit ./yaml-examples/withSequence-example.yaml  --watch 
argo -n argo submit ./yaml-examples/withItems-example.yaml  --watch --log
argo -n argo submit ./yaml-examples/withItems-example-complex.yaml  --watch --log
argo -n argo submit ./yaml-examples/withParam-example-complex.yaml  --watch 
argo -n argo submit ./yaml-examples/withParam-example-more-complex.yaml  --wait
argo -n argo submit ./yaml-examples/coinflip.yaml --watch 
argo -n argo submit ./yaml-examples/coinflip-recursion.yaml --watch 
argo -n argo submit ./yaml-examples/retry-strategy-example.yaml --watch 
argo -n argo submit ./yaml-examples/manage-k8s-resources-example.yaml --watch 
argo -n argo submit ./yaml-examples/daemon-containers-example.yaml --watch 
argo -n argo submit ./yaml-examples/sidecar-containers-example.yaml --watch 
argo -n argo submit ./yaml-examples/dind-containers-example.yaml --watch 
argo -n argo submit ./yaml-examples/variable-reference-example.yaml --watch 
argo -n argo template create ./yaml-examples/loop-test-workflowtemplate.yaml
argo -n argo template create ./yaml-examples/enum-dropdown-wftemplates.yaml
argo -n argo submit --from=wftmpl/loop-test  --wait 
argo template create https://raw.githubusercontent.com/argoproj/argo-workflows/main/examples/workflow-template/templates.yaml
argo submit https://raw.githubusercontent.com/argoproj/argo-workflows/main/examples/workflow-template/hello-world.yaml
# workflow templates can be managed using kubectl apply -f and kubectl get wftmpl
kubectl -n argo apply -f https://raw.githubusercontent.com/argoproj-labs/argo-workflows-catalog/master/templates/hello-world/manifests.yaml
kubectl -n argo apply -f https://raw.githubusercontent.com/argoproj/argo-workflows/main/examples/cluster-workflow-template/clustertemplates.yaml

#argo -n argo submit ./template-deadline-example.yaml --watch 
#argo -n argo submit ./workflow-deadline-example.yaml --watch 
#argo -n argo submit ./exit-handler-example.yaml --watch 
#argo -n argo submit ./suspend-example.yaml --watch 
#argo -n argo submit ./ci-example.yaml --watch 
#argo -n argo submit ./ci-influxdb-example.yaml --watch 

# configure first artifact storage
# https://argo-workflows.readthedocs.io/en/release-3.5/configure-artifact-repository/
argo -n argo submit ./yaml-examples/artifact-passing.yaml  --watch

# customized images
argo -n argo submit ./yaml-examples/automl.yaml --watch --log
argo -n argo submit ./yaml-examples/amlp.yaml --watch --log
argo -n argo submit ./yaml-examples/tsml.yaml --watch --log

# using template
argo -n argo template create ./yaml-examples/combine-workflowtemplate.yaml
argo -n argo submit ./yaml-examples/combine-run-from-workflowtemplate.yaml --watch --log

# demo workflows
argo -n argo submit ./yaml-examples/hello.yaml --watch 
argo -n argo submit ./yaml-examples/sequential-preprocessing.yaml --watch 
argo -n argo submit ./yaml-examples/dag-rust-ml.yaml --watch 
argo -n argo submit ./yaml-examples/parallel.yaml --watch 
argo -n argo submit ./yaml-examples/parallel-combine.yaml --watch 

# cluster template
kubectl -n argo apply -f ./yaml-examples/parallel-cluster-template.yaml 
argo -n argo submit ./yaml-examples/parallel-cluster-template-workflow.yaml --watch

# connect to minio
# edit configmap to access minio
kubectl create secret -n argo generic my-minio-cred --from-literal=access-key='DeF31RQ6JeftPNaQf65Z' --from-literal=secret-key='fFOVIQqecbwfipE8DGTlA4b8h1zmSSLN3hdkS4Ph'

kubectl -n argo edit configmap workflow-controller-configmap
data:
  artifactRepository: |
    archiveLogs: true
    s3:
      bucket: mybucket
      endpoint: minio-tenant-1-hl.minio-tenant-1.svc.cluster.local
      insecure: true
      accessKeySecret:
       name: my-minio-cred
       key: access-key
      secretKeySecret:
       name: my-minio-cred
       key: secret-key

