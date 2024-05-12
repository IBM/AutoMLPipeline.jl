# 1. run kubespray to start a cluster: ./kubespray/install_k8s.sh or 
1. install k0s at ./k0sctl/run.sh

2. enable pvc/pv/storageclass using nfs-provisioner plugin and make it default storage class

kubectl create namespace persistence

helm -n persistence repo add \
   nfs-subdir-external-provisioner https://kubernetes-sigs.github.io/nfs-subdir-external-provisioner/

# for kube1
helm -n persistence install nfs-subdir-external-provisioner nfs-subdir-external-provisioner/nfs-subdir-external-provisioner \
      --set nfs.server=10.242.7.29 --set nfs.path=/mnt/disk2/common

# for mccojhub
helm -n persistence install nfs-subdir-external-provisioner \
   nfs-subdir-external-provisioner/nfs-subdir-external-provisioner \
   --set nfs.server=10.38.225.120 --set nfs.path=/mnt/diskxvdc/nfsprovisioner

# for mccoeu
helm -n persistence install nfs-subdir-external-provisioner \
   nfs-subdir-external-provisioner/nfs-subdir-external-provisioner \
   --set nfs.server=10.242.7.227 --set nfs.path=/mnt/diskxvdc/nfsprovisioner

kubectl -n persistence get pv,pvc,replicaset,deployment
kubectl -n persistence create -f ./nfs-provisioner/test-claim.yaml -f ./nfs-provisioner/test-pod.yaml
kubectl -n persistence get pv,pvc,replicaset,deployment
kubectl -n persistence create -f ./nfs-provisioner/test-another-pod.yaml
kubectl -n persistence delete -f ./nfs-provisioner/test-pod.yaml -f ./nfs-provisioner/test-another-pod.yaml -f ./nfs-provisioner/test-claim.yaml

3. create as default storageclass
kubectl patch storageclass nfs-client -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
kb -n persistence get storageclass

4. install jupyterhub (optional)
# install jupyterhub
kubectl create namespace jhub

helm repo add jupyterhub https://hub.jupyter.org/helm-chart/
helm repo update
helm search repo jupyterhub/jupyterhub
helm pull  jupyterhub/jupyterhub

helm upgrade \
   --install jhubr jupyterhub/jupyterhub \
   --namespace jhub \
   --create-namespace \
   --version="3.2.1" \
   --values values.yaml

# delete jupyterhub
# helm -n jhub delete jhubr

5. install prometheus (must be done in the remote control-plane)
   - cd kube-devsandbox
   - change storageclass to nfs-client

# ssh to mccoeu1
ssh root@mccoeu1.sl.cloud9.ibm.com
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
kubectl create ns prometheus 
#helm install prometheus-stack prometheus-community/kube-prometheus-stack --values ./manifests/monitoring/prom-config.yml -n prometheus
helm pull prometheus-community/kube-prometheus-stack
# edit ./kube-prometheus-stack/values.yaml
helm install prometheus-stack prometheus-community/kube-prometheus-stack --values ./kube-prometheus-stack/values.yaml -n prometheus

# validate
kubectl --namespace prometheus get pods -l "release=prometheus-stack"
kubectl -n prometheus patch svc prometheus-stack-kube-prom-prometheus  -p '{"spec": {"type": "NodePort"}}'
kubectl -n prometheus patch svc prometheus-stack-grafana  -p '{"spec": {"type": "NodePort"}}'
kubectl -n prometheus patch svc prometheus-stack-kube-state-metrics  -p '{"spec": {"type": "NodePort"}}'

7. install influxdb
kb create ns influxdb2
helm repo add influxdata https://helm.influxdata.com
helm search repo influxdata

helm -n influxdb2 upgrade --install influxdb2 \
  --set persistence.enabled=true,persistence.size=200Gi\
  influxdata/influxdb2

# make it default
kubectl patch storageclass nfs-client -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
kubectl get storageclass

kb get svc -n influxdb2
kubectl -n influxdb2 patch svc influxdb2  -p '{"spec": {"type": "NodePort"}}'

echo $(kubectl get secret influxdb2-auth -o "jsonpath={.data['admin-password']}" --namespace influxdb2 | base64 --decode)

kb -n influxdb2 get svc
kb -n influxdb2 get all

prom-scraper: http://mccoeu1.sl.cloud9.ibm.com:31194/metrics

#helm -n influxdb2 delete  influxdb2  
#kb delete ns influxdb2  

# telegraf to remote write to influxdb from prometheus
https://www.influxdata.com/blog/prometheus-remote-write-support-with-influxdb-2-0/

#export INFLUX_TOKEN=G0w5c7PiGgyOIV1-xEXaGZd5f2WDs2QaNnkWLu3x57_xQjUZd72xAVG7hKJnXpxwEPPLYNruhp1gyq8wR_lcTg==
#telegraf --config http://mccoeu1.sl.cloud9.ibm.com:30958/api/v2/telegrafs/0b2dbf9cbcd27000

8. Install the Latest Telegraf

You can install the latest Telegraf by visiting the InfluxData Downloads page. If you already have Telegraf installed on your system, make sure it's up to date. You will need version 1.9.2 or higher.
https://portal.influxdata.com/downloads/

wget -q https://repos.influxdata.com/influxdata-archive_compat.key
echo '393e8779c89ac8d958f81f942f9ad7fb82a25e133faddaf92e15b16e6ac9ce4c influxdata-archive_compat.key' | sha256sum -c && cat influxdata-archive_compat.key | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg > /dev/null
echo 'deb [signed-by=/etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg] https://repos.influxdata.com/debian stable main' | sudo tee /etc/apt/sources.list.d/influxdata.list

sudo apt-get update && sudo apt-get install telegraf

8.1 Configure your API Token

Your API token is required for pushing data into InfluxDB. You can copy the following command to your terminal window to set an environment variable with your API token.

# for mccoeu1

ssh root@mccoeu1.sl.cloud9.ibm.com
download the config from influxdb telegraf and copy to /etc/telegraf/telegraf.d

export INFLUX_TOKEN=KuIqTk01Z93vJiWD8gUWJfodt3P5n9ihvE107pnt6-Nmm4gX4ww3w_vMdfdaGbznQCar1RkQhelcs6iBqhepLg==
telegraf --config /etc/telegraf/telegraf.conf --config-directory /etc/telegraf/telegraf.d 

8.2 Start Telegraf: for influxdb2 instruction

# for mccoeu1
# download the created telegraf config from influxdb and copy to telegraf.d/
ssh mcceoeu1.sl.cloud9.ibm.com
vi /etc/telegraf/telegraf.d/prom-telegraf.conf

 [[inputs.prometheus]]
  urls = ["http://mccoeu1.sl.cloud9.ibm.com:31143/api/cart/metrics", "http://mccoeu1.sl.cloud9.ibm.com:31143/api/payment/metrics","http://mccoeu1.sl.cloud9.ibm.com:32496/metrics"]
  metric_version = 2

 [[outputs.influxdb_v2]]
  urls = ["http://mccoeu1.sl.cloud9.ibm.com:30958"]
  token = "G0w5c7PiGgyOIV1-xEXaGZd5f2WDs2QaNnkWLu3x57_xQjUZd72xAVG7hKJnXpxwEPPLYNruhp1gyq8wR_lcTg=="
  organization = "influxdata"
  bucket = "prometheus"

Finally, you can run the following command to start the Telegraf agent running on your machine.

ssh mcceoeu1.sl.cloud9.ibm.com
nohup telegraf --config /etc/telegraf/telegraf.conf --config-directory /etc/telegraf/telegraf.d &

#telegraf --config http://mccoeu1.sl.cloud9.ibm.com:30958/api/v2/telegrafs/0b2dbf9cbcd27000

# for mccosmall
# download the created telegraf config from influxdb and copy to telegraf.d/
export INFLUX_TOKEN=3Qbra2TGkkW09GMwXn06QQox8ZdUoCexwvYqMeGvtqtUpBiVP6uoQ_7EkZcZ_nw9O507F8Y3dDahWZd9ujBf-A==

 [[inputs.prometheus]]
  urls = ["http://mccosmall1.sl.cloud9.ibm.com:32196/api/cart/metrics", "http://mccosmall1.sl.cloud9.ibm.com:32196/api/payment/metrics","http://mccosmall1.sl.cloud9.ibm.com:32509/metrics"]
  metric_version = 2

[[outputs.influxdb_v2]]
urls = ["http://mccosmall1.sl.cloud9.ibm.com:32347"]
token = "$INFLUX_TOKEN"
organization = "influxdata"
bucket = "prometheus"

ssh mcceoeu1.sl.cloud9.ibm.com
nohup telegraf --config /etc/telegraf/telegraf.conf --config-directory /etc/telegraf/telegraf.d &

# 8. update prometheus values.yaml: old influxdb instruction
# 
# # ssh to mccoeu1
# ssh mccoeu1.sl.clou9.ibm.com
# 
# add these:  this is old version of influxdb
# 
#   scrape_configs:
#     - job_name: 'prometheus'
#       static_configs:
#       - targets: ['http://mccoeu1.sl.cloud9.ibm.com:31025']
#   
#   remote_write:
#     - url: "http://mccoeu1.sl.cloud9.ibm.com:8080/receive"
# 
# helm upgrade --install prometheus-stack prometheus-community/kube-prometheus-stack --values ./manifests/monitoring/prom-config.yml -n observability
