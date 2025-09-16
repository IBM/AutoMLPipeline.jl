# ============
# Insructions
# ============
# - change the IP addresses to your machines addresses
# - make sure you can ssh as root to the machines without the need of password
#   by exporting your public key using ssh-copy-id to the remote machines
# - more info: https://github.com/alexellis/k3sup
#
[ ! -f /usr/local/bin/k3sup ] && curl -sLS https://get.k3sup.dev | sh && sudo install ./k3sup* /usr/local/bin/k3sup

PRIMARYSERVER=x.x.x.x
WORKERS=(y.y.y.y z.z.z.z a.a.a.a)

echo "Setting up primary server 1"
k3sup install --host $PRIMARYSERVER \
	--user root \
	--cluster \
	--local-path kubeconfig \
	--context default

echo "Fetching the server's node-token into memory"
export NODE_TOKEN=$(k3sup node-token --host $PRIMARYSERVER --user root)

for worker in $WORKERS; do
	echo "Setting up worker: $worker"
	k3sup join --host $worker --server-host $PRIMARYSERVER --node-token "$NODE_TOKEN" --user root &
done

mkdir -p $HOME/.kube
cp kubeconfig ~/.kube/k3s.config
export KUBECONFIG=$KUBECONFIG:$HOME/.kube/k3s.config
echo 'export KUBECONFIG=$KUBECONFIG:$HOME/.kube/k3s.config' >>$HOME/.zshrc
echo 'export KUBECONFIG=$KUBECONFIG:$HOME/.kube/k3s.config' >>$HOME/.bashrc
