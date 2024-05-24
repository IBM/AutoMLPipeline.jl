# https://blog.min.io/deploy-minio-with-argocd-in-kubernetes/amp/

#-- Install argocd
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
#
# use port-forward to login
#> kubectl -n argocd port-forward svc/argocd-server 8080:80
#> argocd login localhost:8080
#
# or use loadbalancer to login
kubectl -n argocd patch svc argocd-server  -p '{"spec": {"type": "LoadBalancer"}}'
#
argo_ip=$(kubectl -n argocd get svc | grep argocd-server | grep LoadBalancer|tr -s " " ","|cut -d"," -f4)
initial_password=$(argocd admin initial-password -n argocd)
current_context=$(kubectl config current-context)
argocd login ${argo_ip} --username admin --password ${initial_password} --insecure
argocd cluster add ${current_context} --in-cluster --upsert
#
# and update the password
argocd account update-password

#-- Create minio-operator
kubectl create namespace minio-operator
argocd app create minio-operator \
   --repo https://github.com/cniackz/minio-argocd.git \
   --path minio-operator \
   --dest-namespace minio-operator \
   --dest-server https://kubernetes.default.svc --insecure --upsert

#-- Login to the svc by LoadBalancer or NodePort
# source: https://min.io/docs/minio/kubernetes/upstream/operations/install-deploy-manage/minio-operator-console.html
kubectl -n minio-operator patch svc console  -p '{"spec": {"type": "LoadBalancer"}}'
jwtoken=$(kubectl get secret/console-sa-secret -n minio-operator -o json | jq -r '.data.token' | base64 -d)
# or jwtoken=$(kubectl -n minio-operator get secret console-sa-secret -o jsonpath='{.data.token}' | base64 --decode)
minio_operator=$(kubectl -n minio-operator get svc| grep console | grep LoadBalancer|tr -s " " ","|cut -d"," -f4)
open  $(echo http://${minio_operator}:9090)
echo $jwtoken 
#
# or login by proxy
#> kubectl minio proxy
#> open http://localhost:9090

#-- Create a tenant via argocd
kubectl create namespace minio-tenant
argocd app create minio-tenant \
   --repo https://github.com/cniackz/minio-argocd.git \
   --path minio-tenant \
   --dest-namespace minio-tenant \
   --dest-server https://kubernetes.default.svc \
   --insecure \
   --upsert
#
# Note: this is the orriginal helm install of tenant outside argocd
#> helm search repo minio
#> helm install --namespace minio-tenant \
#>   --create-namespace tenant minio-operator/tenant

#-- Access tenant console
#
# using port-forward
#> kubectl --namespace minio-tenant port-forward svc/myminio-console 9443:9443
#
# or using LoadBalancer to access
kubectl -n minio-tenant patch svc myminio-console  -p '{"spec": {"type": "LoadBalancer"}}'
minio_tenant=$(kubectl -n minio-tenant get svc| grep myminio-console | grep LoadBalancer|tr -s " " ","|cut -d"," -f4)
#
# values.yaml info of password 
# secrets:
#   name: myminio-env-configuration
#   # MinIO root user and password
#   accessKey: minio
#   secretKey: minio123
# ## MinIO Tenant Definition
# info from https://github.com/cniackz/minio-argocd/blob/main/minio-tenant/values.yaml
# user: minio, password: minio123
#
export MINIO_ROOT_USER="minio"
export MINIO_ROOT_PASSWORD="minio123"
open $(echo https://${minio_tenant}:9443)


# Setting up Artifact Bucket Credentials
#
# connect argo with minio: https://medium.com/@michael.cook.talk/argo-workflows-minio-nginx-8911b988b5c8
# create bucket in minio console
# create Access Key in minio console
# Access Key: FxcWDBqCHApF1HuF
echo -n FxcWDBqCHApF1HuF | base64
# Secret Key: DshXCJ42dlSq6PgwTxCEwkRB09nywo3K
echo -n DshXCJ42dlSq6PgwTxCEwkRB09nywo3K | base64
# write these base64 values into artifact-bucket-credential.yaml

ACCESS_KEY=FxcWDBqCHApF1HuF
SECRET_KEY=DshXCJ42dlSq6PgwTxCEwkRB09nywo3K

# Add access credential so that argo workflow can connect to COS
kubectl -n argo apply -f  artifact-bucket-credential.yaml
kubectl create secret -n argo generic my-minio-cred --from-literal=access-key=$ACCESS_KEY --from-literal=secret-key=$SECRET_KEY

# edit configmap to add access credentials
kubectl -n argo get configmap 
kubectl -n argo edit configmap workflow-controller-configmap
data:
  artifactRepository: |
    archiveLogs: true
    s3:
      bucket: my-bucket
      endpoint: argo-artifacts.default:9000
      insecure: true
      accessKeySecret:
       name: my-minio-cred
       key: access-key
      secretKeySecret:
       name: my-minio-cred
       key: secret-key
