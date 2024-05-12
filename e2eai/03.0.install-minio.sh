# https://github.com/minio/operator/issues/70
# https://min.io/docs/minio/kubernetes/upstream/operations/install-deploy-manage/deploy-minio-tenant.html#minio-k8s-deploy-minio-tenant

kubectl krew update
kubectl krew install minio

kubectl minio init
# to delete: kubectl minio delete

kubectl -n minio-operator get all

kubectl -n minio-operator patch service console -p '{"spec": {"type": "LoadBalancer"}}' 

# get the JWT credential
kubectl minio proxy -n minio-operator

# Create a New Tenant
# The following kubectl minio command creates a MinIO Tenant with 4 nodes, 
# 16 volumes, and a total capacity of 16Ti. 
# This configuration requires at least 16 Persistent Volumes.

kubectl create ns minio-tenant-1
kubectl minio tenant create minio-tenant-1 \
--servers 2 \
--volumes 4 \
--capacity 10Gi \
--namespace minio-tenant-1 \
--storage-class nfs-client

# Username: DD1D57328QATUFSLFXTZ
# Password: FJ9C2F3iX7H5KfzWxVfYKweNQt09dE4718JXAuB6
# Note: Copy the credentials to a secure location. MinIO will not display these again.
# MINIO_ROOT_PASSWORD: v4draVI8nnlVxVcxaaaZ4mABQ5UtixvtXAtpxR2W
# MINIO_ROOT_USER: 3L633U8E4IE8MY1H9H22

kubectl -n minio-tenant-1 patch svc minio-tenant-1-console  -p '{"spec": {"type": "LoadBalancer"}}'

kubectl -n minio-operator patch svc console  -p '{"spec": {"type": "LoadBalancer"}}'

kubectl minio proxy

# credentials are generated in the console and saved in ./minio/ directory
