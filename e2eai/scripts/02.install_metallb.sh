# refs: https://metallb.universe.tf/
# kubectl edit configmap -n kube-system kube-proxy

# ClusterIP vs NodePort vs LoadBalancer
#  - every pod has a private IP
#  - a service is a static IP 
#  - ClusterIP is an internal service
#  - NodePort port-forward from outside to ClusterIP service
#  - a Loadbalancer performs dynamic port-forwarding from outside to ClusterIP service 

kubectl create ns metallb-system 

# install by manifest
kubectl apply -f metallb.yaml

# install by helm
# helm repo add metallb https://metallb.github.io/metallb
# helm install metallb metallb/metallb
 
# configure using layer 2
# addresses based on the cluster nodes network range

cat <<EOF | kubectl create -f -
apiVersion: v1
kind: ConfigMap
metadata:
  namespace: metallb-system
  name: config
data:
  config: |
    address-pools:
    - name: default
      protocol: layer2
      addresses:
      - 9.4.190.230-9.4.190.250
EOF

# expose grafana dashboard with LoadBalancer
gs=$(kubectl -n prometheus get svc -A | grep "grafana " | tr -s " " " " |cut -d" " -f2)
kubectl -n prometheus patch service $gs -p '{"spec": {"type": "LoadBalancer"}}'

# verify grafana is exposed by LoadBalancer
kubectl get svc -A | grep grafana

# create deployment and expose the service using LoadBalancer
kubectl create namespace web
kubectl -n web create deployment nginx --image nginx
kubectl -n web get all -o wide
kubectl -n web expose deployment nginx --port 80 --type LoadBalancer
kubectl -n web get all -o wide

# differentiate between ClusterIP, NodePort, LoadBalancer
kubectl get services -A

# cleanup
kubectl delete namespace web
kubectl delete namespace metallb-system
