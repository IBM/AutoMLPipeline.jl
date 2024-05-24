# cd to github repo
cd argo-workflow/argo-workflows
git checkout v3.5.5
cd manifests
# install minio, postgres, argo-server, httpbin
kubectl -n argo apply -f ./quick-start-minimal.yaml
kubectl -n argo delete -f argo-workflow/argo-workflows/manifests/quick-start-minimal.yaml

# create default service account
kubectl -n argo create rolebinding default-admin --clusterrole=admin --serviceaccount=argo:default

# for load balancer
kubectl -n argo patch svc argo-server  -p '{"spec": {"type": "LoadBalancer"}}'
kubectl -n argo patch svc minio  -p '{"spec": {"type": "LoadBalancer"}}'
kubectl -n argo patch svc httpbin  -p '{"spec": {"type": "LoadBalancer"}}'
export MINIO=$(kubectl -n argo get svc |grep minio|tr -s " " " " |cut -d" " -f4)
export ARGO_SERVER=$(kubectl -n argo get svc |grep argo-server|tr -s " " " " |cut -d" " -f4)
open "https://${ARGO_SERVER}:2746"
open "http://${MINIO}:9001" 

# for nodeport
kubectl -n argo patch svc argo-server  -p '{"spec": {"type": "NodePort"}}'
kubectl -n argo patch svc minio  -p '{"spec": {"type": "NodePort"}}'
kubectl -n argo patch svc httpbin  -p '{"spec": {"type": "NodePort"}}'
export NODE=mccoeu1.sl.cloud9.ibm.com
export MINIO=http://mccoeu1.sl.cloud9.ibm.com
export MINIO_PORT=$(kubectl -n argo get svc |grep minio|tr -s "[ /]" ":" |cut -d":" -f8)
export ARGO_PORT=$(kubectl -n argo get svc |grep argo-server|tr -s "[ /]" ":" |cut -d":" -f6)
export MINIO_PORT=30594
open "http://${NODE}:${MINIO_PORT}"
open "https://${NODE}:${ARGO_PORT}" 


#kubectl -n argo port-forward svc/argo-server 2746:2746
#export ARGO_SECURE=true
#argo -n argo server --auth-mode=server

# delete all workflows
argo -n argo delete --all

# examples
argo -n argo submit  ./yaml-examples/hello1.yaml 
argo -n argo submit ./yaml-examples/hello1.yaml -p message="hello universe"  
argo -n argo submit ./yaml-examples/hello2.yaml  
argo -n argo submit ./yaml-examples/dag-parallel.yaml  
argo -n argo submit ./yaml-examples/run-scripts.yaml  
argo -n argo submit ./yaml-examples/output-parameters.yaml  

# secret
kubectl -n argo create secret generic my-secret --from-literal=mypassword=S00perS3cretPa55word
argo -n argo submit ./yaml-examples/secret-example.yaml  
argo -n argo submit --serviceaccount argo-sa ./yaml-examples/argo-workflow-example-volumeclaim1.yaml 

# create pvc first called my-existing-volume
kubectl -n argo apply -f ./yaml-examples/pvc.yaml
argo -n argo submit --serviceaccount argo-sa ./yaml-examples/argo-workflow-example-volumeclaim2.yaml 
#kubectl -n argo delete -f ./yaml-examples/pvc.yaml

# template level
argo -n argo submit ./yaml-examples/argo-workflow-example-volumeclaim3.yaml 
#argo -n argo logs  template-level-volume-kzwm6

# sequence example
argo -n argo submit ./yaml-examples/withSequence-example.yaml   
argo -n argo submit ./yaml-examples/withItems-example.yaml   
argo -n argo submit ./yaml-examples/withItems-example-complex.yaml   
argo -n argo submit ./yaml-examples/withParam-example-complex.yaml   
argo -n argo submit ./yaml-examples/withParam-example-more-complex.yaml
argo -n argo submit ./yaml-examples/coinflip.yaml  
argo -n argo submit ./yaml-examples/coinflip-recursion.yaml  
argo -n argo submit ./yaml-examples/retry-strategy-example.yaml  
argo -n argo submit ./yaml-examples/manage-k8s-resources-example.yaml  
argo -n argo submit ./yaml-examples/daemon-containers-example.yaml  
argo -n argo submit ./yaml-examples/sidecar-containers-example.yaml  
argo -n argo submit ./yaml-examples/dind-containers-example.yaml  

argo -n argo submit ./yaml-examples/variable-reference-example.yaml  
argo -n argo template create ./yaml-examples/loop-test-workflowtemplate.yaml
argo -n argo template create ./yaml-examples/enum-dropdown-wftemplates.yaml
argo -n argo submit --from=wftmpl/loop-test   
argo template create https://raw.githubusercontent.com/argoproj/argo-workflows/main/examples/workflow-template/templates.yaml
argo submit https://raw.githubusercontent.com/argoproj/argo-workflows/main/examples/workflow-template/hello-world.yaml

# workflow templates can be managed using kubectl apply -f and kubectl get wftmpl
kubectl -n argo apply -f https://raw.githubusercontent.com/argoproj-labs/argo-workflows-catalog/master/templates/hello-world/manifests.yaml
kubectl -n argo apply -f https://raw.githubusercontent.com/argoproj/argo-workflows/main/examples/cluster-workflow-template/clustertemplates.yaml

#argo -n argo submit ./template-deadline-example.yaml  
#argo -n argo submit ./workflow-deadline-example.yaml  
#argo -n argo submit ./exit-handler-example.yaml  
#argo -n argo submit ./suspend-example.yaml  
#argo -n argo submit ./ci-example.yaml  
#argo -n argo submit ./ci-influxdb-example.yaml  

# configure first artifact storage
# https://argo-workflows.readthedocs.io/en/release-3.5/configure-artifact-repository/
argo -n argo submit ./yaml-examples/artifact-passing.yaml  

# customized images
argo -n argo submit ./yaml-examples/automl.yaml  
argo -n argo submit ./yaml-examples/amlp.yaml  
argo -n argo submit ./yaml-examples/tsml.yaml  

# using template
argo -n argo template create ./yaml-examples/combine-workflowtemplate.yaml
argo -n argo submit ./yaml-examples/combine-run-from-workflowtemplate.yaml  

# demo workflows
argo -n argo submit ./yaml-examples/hello.yaml  
argo -n argo submit ./yaml-examples/sequential-preprocessing.yaml  
argo -n argo submit ./yaml-examples/dag-rust-ml.yaml  
argo -n argo submit ./yaml-examples/parallel.yaml  
argo -n argo submit ./yaml-examples/parallel-combine.yaml  

# cluster template
kubectl -n argo apply -f ./yaml-examples/parallel-cluster-template.yaml 
argo -n argo submit ./yaml-examples/parallel-cluster-template-workflow.yaml 

