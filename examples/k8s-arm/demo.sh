kubectl create namespace julia

# create role and role-bindings
kubectl -n julia create -f permissions.yaml

# deploy
kubectl -n julia create -f automlpipeline.yaml

# follow logs of main
kubectl -n julia logs automlpipeline-k7mb5 -f

# delete namespace and resources
kubectl delete namespace julia
