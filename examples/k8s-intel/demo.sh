# check cluster health
kubectl get nodes

# list namespaces
kubectl get ns

# create julia namespace 
# for deploying julia applications
kubectl create ns julia

# verify/edit permissions
vi permissions.yaml
kubectl -n julia create -f permissions.yaml

# verify/edit deployment
vi automlpipeline.yaml
kubectl -n julia create -f automlpipeline.yaml

# verify pod creation
watch kubectl -n julia get pods

# follow logs of main orchestrator
mj=$(k -n julia get pods |grep -v NAME | head -1 |tr -s " " " " | cut -d" " -f1)
kubectl -n julia logs $mj -f

# delete namespace and resources
kubectl delete ns julia
