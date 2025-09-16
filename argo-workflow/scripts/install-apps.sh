# helm apps
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
# helm repo add influxdata https://helm.influxdata.com
helm repo add grafana https://grafana.github.io/helm-charts
helm repo add open-telemetry https://open-telemetry.github.io/opentelemetry-helm-charts
helm repo add traefik https://traefik.github.io/charts
helm repo add community-charts https://community-charts.github.io/helm-charts
helm repo add minio-helm https://charts.min.io/
helm add repo bitnami https://charts.bitnami.com/bitnami
helm repo add cowboysysop https://cowboysysop.github.io/charts/
helm repo update

# helm install my-release oci://ghcr.io/cowboysysop/charts/flowise
helm install flowise cowboysysop/flowise -n flowise --create-namespace

## create default storageclass
kubectl patch storageclass nfs-client -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
kubectl patch storageclass local-path -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"false"}}}'

# argo
myNameSpaceArgo=argo
kubectl get namespace | grep -q "^$myNameSpaceArgo " || kubectl create namespace $myNameSpaceArgo
kubectl -n $myNameSpaceArgo apply -f ./quick-start-minimal-v3.6.5.yaml
kubectl -n $myNameSpaceArgo create rolebinding default-admin --clusterrole=admin --serviceaccount=argo:default
kubectl -n argo patch svc argo-server -p '{"spec": {"type": "LoadBalancer"}}'

# disable https and security by server mode
kubectl patch deployment \
	argo-server \
	--namespace argo \
	--type='json' \
	-p='[{"op": "replace", "path": "/spec/template/spec/containers/0/args", "value": [
  "server",
  "--auth-mode=server",
  "--secure=false"
]},
{"op": "replace", "path": "/spec/template/spec/containers/0/readinessProbe/httpGet/scheme", "value": "HTTP"}
]'

# loki
helm -n loki install loki-stack grafana/loki-stack \
	--values loki-values.yaml --create-namespace

# argocd
myNamespaceArgoCD=argocd
kubectl get namespace | grep -q "^$myNamespaceArgoCD " || kubectl create namespace $myNamespaceArgoCD
kubectl -n $myNamespaceArgoCD apply -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
kubectl -n argocd patch svc argocd-server -p '{"spec": {"type": "NodePort"}}'

# mlflow
# take note to delete pvc when reinstalling
# change default port 80 to different values for LoadBalancer
helm -n mlflow upgrade --install sunrise bitnami/mlflow \
	--version 3.0.0 --create-namespace --set minio.persistence.size=50Gi --set tracking.service.ports.http=5080 --set tracking.service.ports.https=50443 --set minio.service.ports.api=5081 --set tracking.auth.enabled=false

ROOT_USER="username"
ROOT_PASSWORD="yourpassword"
export ENDPOINT="minio.minio:9000" BUCKET=thanos

## minio
# https://github.com/minio/minio/blob/master/helm/minio/README.md
helm -n minio install minio \
	--set replicas=2 \
	--set persistence.size=50Gi \
	--set rootUser=${ROOT_USER},rootPassword=${ROOT_PASSWORD} \
	--set persistence.enabled=true \
	minio-helm/minio --create-namespace

kubectl -n minio patch svc minio-console -p '{"spec": {"type": "LoadBalancer"}}'

kubectl run --namespace minio \
	minio-client --rm --tty -i --restart='Never' \
	--env MINIO_SERVER_ROOT_USER=${ROOT_USER} \
	--env MINIO_SERVER_ROOT_PASSWORD=${ROOT_PASSWORD} \
	--env MINIO_SERVER_HOST=minio.minio \
	--image docker.io/bitnami/minio-client -- mc mb -p minio/thanos

cat >objstore.yml <<EOF
type: S3
config:
  endpoint: "${ENDPOINT}"
  bucket: "${BUCKET}"
  access_key: "${ROOT_USER}"
  secret_key: "${ROOT_PASSWORD}"
  insecure: true
EOF

kubectl create secret generic thanos-objstore \
	--from-file=objstore.yml -o yaml \
	--dry-run=client | kubectl -n prometheus apply -f -

helm -n prometheus upgrade --install prometheus-stack prometheus-community/kube-prometheus-stack \
	--set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
	--set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
	--set "prometheus.prometheusSpec.enableFeatures[0]=otlp-write-receiver" \
	--set alertmanager.ingress.enabled=true \
	--set prometheus.enabled=true \
	--values ./prometheus-with-thanos-values.yaml \
	--create-namespace

# install other thanos components
helm -n prometheus upgrade --install thanos \
	bitnami/thanos --values ./thanos-values.yaml \
	--create-namespace

# install kserve
# https://kserve.github.io/website/latest/admin/kubernetes_deployment/#2-install-network-controller
arkade install cert-manager

#gateway
kubectl -n kserve apply -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.2.1/standard-install.yaml

kubectl -n kserve apply -f - <<EOF
apiVersion: gateway.networking.k8s.io/v1
kind: GatewayClass
metadata:
  name: envoy
spec:
  controllerName: gateway.envoyproxy.io/gatewayclass-controller
EOF

kubectl -n kserve apply -f - <<EOF
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: kserve-ingress-gateway
  namespace: kserve
spec:
  gatewayClassName: envoy
  listeners:
    - name: http
      protocol: HTTP
      port: 80
      allowedRoutes:
        namespaces:
          from: All
    - name: https
      protocol: HTTPS
      port: 443
      tls:
        mode: Terminate
        certificateRefs:
          - kind: Secret
            name: my-secret
            namespace: kserve
      allowedRoutes:
        namespaces:
          from: All
  infrastructure:
EOF

helm -n kserve install kserve-crd oci://ghcr.io/kserve/charts/kserve-crd --version v0.15.0

helm -n kserve install kserve oci://ghcr.io/kserve/charts/kserve --version v0.15.0 \
	--set kserve.controller.deploymentMode=RawDeployment \
	--set kserve.controller.gateway.ingressGateway.enableGatewayApi=true --set kserve.controller.gateway.ingressGateway.kserveGateway=kserve/kserve-ingress-gateway

# install coroot observability tool
helm repo add coroot https://coroot.github.io/helm-charts
helm repo update coroot

helm install -n coroot --create-namespace coroot-operator coroot/coroot-operator

helm install -n coroot coroot coroot/coroot-ce

kubectl port-forward -n coroot service/coroot-coroot 8080:8080

helm uninstall coroot -n coroot
helm uninstall coroot-operator -n coroot
