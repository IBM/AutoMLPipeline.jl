helm repo add minio-helm https://charts.min.io/
helm repo update

#> helm install --namespace minio --create-namespace --set rootUser=rootuser,rootPassword=rootpass123 --generate-name minio-helm/minio

helm install --namespace minio --create-namespace \
   --set resources.requests.memory=512Mi --set replicas=1 \
   --set persistence.enabled=false --set mode=standalone \
   --generate-name minio-helm/minio

rootPassword=$(kubectl get secret --namespace minio myminio -o jsonpath="{.data.rootPassword}")
rootUser=$(kubectl get secret --namespace minio myminio -o jsonpath="{.data.rootUser}")

kubectl -n argo edit configmap workflow-controller-configmap
kubectl -n argo edit configmap argo-worklows-argo-workflows-workflow-controller-configmap
