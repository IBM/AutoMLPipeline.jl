# export  DOCKER_HOST=/var/run/docker.sock
# kind-podman

podman machine init --cpus 4 --memory 12288
podman machine set --rootful
podman machine start

cat > kind-config.yaml <<EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
- role: worker
- role: worker
EOF

kind create cluster --name kindk8s --config kind-config.yaml 
kubectl cluster-info --context kind-kindk8s
kubectx kind-kindk8s
kind get kubeconfig --name kindk8s > ~/.kube/kind.config

export KUBECONFIG=$KUBECONFIG:~/.kube/kind.config
[ -f $HOME/.zhsrc ] && echo 'export KUBECONFIG=$KUBECONFIG:$HOME/.kube/kind.config' >> $HOME/.zshrc
[ -f $HOME/.bashrc ] && echo 'export KUBECONFIG=$KUBECONFIG:$HOME/.kube/kind.config' >> $HOME/.bashrc

# podman container pause -a
# podman container unpause -a

# podman machine stop
# podman machine rm
