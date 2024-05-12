# https://github.com/minio/minio/tree/master/helm/minio

#### this is very buggy workflow
helm repo add minio https://charts.min.io/
helm repo update

helm -n minio install myminio \
   --set persistence.size=5Gi \
   --create-namespace \
   --set replicas=2 \
   --set rootUser=rootuser,rootPassword=rootpass123 minio/minio

### buggy
## helm -n minio install myminio\
##    --set resources.requests.memory=512Mi \
##    --set replicas=1 --set persistence.enabled=false \
##    --create-namespace \
##    --set mode=standalone \
##    --set rootUser=rootuser,rootPassword=rootpass123 minio/minio

kubectl -n minio patch service myminio-console -p '{"spec": {"type": "LoadBalancer"}}'
kubectl -n minio patch service myminio -p '{"spec": {"type": "LoadBalancer"}}'

helm -n minio status myminio
mc mb myminio-local/my-bucket
mc ls myminio-local

helm -n minio uninstall myminio

# helm install --set policies[0].name=mypolicy,\
#    policies[0].statements[0].resources[0]='arn:aws:s3:::bucket1',\
#    policies[0].statements[0].actions[0]='s3:ListBucket',\
#    policies[0].statements[0].actions[1]='s3:GetObject' minio/minio

# helm install --set users[0].accessKey=accessKey,users[0].secretKey=secretKey,\
#    users[0].policy=none,users[1].accessKey=accessKey2,\
#    users[1].secretRef=existingSecret,users[1].secretKey=password,\
#    users[1].policy=none minio/minio

# helm install --set svcaccts[0].accessKey=accessKey,\
#    svcaccts[0].secretKey=secretKey,\
#    svcaccts[0].user=parentUser,svcaccts[1].accessKey=accessKey2,\
#    svcaccts[1].secretRef=existingSecret,svcaccts[1].secretKey=password,\
#    svcaccts[1].user=parentUser2 minio/minio
