export ARGO_SERVER=$(kubectl -n argo get svc |grep argo-server|tr -s " " " " |cut -d" " -f4)
export MINIO=$(kubectl -n argo get svc |grep minio|tr -s " " " " |cut -d" " -f4)
open "https://${ARGO_SERVER}:2746"
open "http://${MINIO}:9001"

argo -n argo submit ./yaml-examples/hello.yaml --watch 

argo -n argo submit ./yaml-examples/sequential-preprocessing.yaml --watch 

argo -n argo submit ./yaml-examples/dag-rust-ml.yaml --watch 

argo -n argo submit ./yaml-examples/parallel.yaml --watch 

argo -n argo submit ./yaml-examples/parallel-combine.yaml --watch 

# cluster template
kubectl -n argo apply -f ./parallel-cluster-template.yaml 
argo -n argo submit ./parallel-cluster-template-workflow.yaml --watch

# configure first artifact storage
# https://argo-workflows.readthedocs.io/en/release-3.5/configure-artifact-repository/
argo -n argo submit ./yaml-examples/artifact-passing.yaml --watch

argo -n argo submit ./yaml-examples/list_buckets.yaml --watch

argo -n argo submit ./yaml-examples/amlp_artifacts.yaml --watch --log

# create pvc first called my-existing-volume
kubectl -n argo apply -f ./yaml-examples/pvc.yaml
argo -n argo submit --serviceaccount argo-sa ./yaml-examples/argo-workflow-example-volumeclaim2.yaml
#kubectl -n argo delete -f ./yaml-examples/pvc.yaml

#mirrord exec -f ./mirrord.json -- /bin/bash 

argo -n argo submit ./yaml-artifacts/key-amlp.yaml --watch --log
argo -n argo submit ./yaml-artifacts/key-amlp-argparse.yaml --watch 
argo -n argo submit ./yaml-artifacts/container-set-amlp.yaml --watch --log
argo -n argo submit ./yaml-artifacts/key-artifacts.yaml --watch

argo -n argo submit ./yaml-artifacts/key-amlp-argparse1.yaml --watch --log
argo -n argo submit ./yaml-artifacts/key-amlp-argparse2.yaml --watch --log

argo -n argo submit ./yaml-artifacts/key-amlp-argparse3.yaml \
   -p workers=3 -p complexity=high -p input=iris.csv \
   -p predictiontype=classification --watch --log


# https://argo-workflows.readthedocs.io/en/latest/cluster-workflow-templates/
kubectl apply -f key-amlp-argpassing-template.yaml 

argo -n argo submit --from clusterworkflowtemplate/argpassing-template \
   -p workers=3 -p complexity=high -p input=iris.csv \
   -p predictiontype=classification --watch --log

argo -n argo submit --from clusterworkflowtemplate/argpassing-template \
   -p workers=3 -p complexity=high -p input=iris_reg.csv \
   -p predictiontype=regression --watch --log
