# instructions
# ============
# - edit k0sctl.yaml to specify the IP addresses of kubernetes master and workers
# - make sure you ssh-copy-id to root@eachmachines your public key to have passwordless root access

k0sctl apply --config k0sctl.yaml
k0sctl kubeconfig --config ./k0sctl.yaml > $HOME/.kube/k0s.config
export KUBECONFIG=$KUBECONFIG:$HOME/.kube/k0s.config
[ -f $HOME/.bashrc ] && echo 'export KUBECONFIG=$KUBECONFIG:$HOME/.kube/k0s.config' >> $HOME/.bashrc
[ -f $HOME/.zshrc ] && echo 'export KUBECONFIG=$KUBECONFIG:$HOME/.kube/k0s.config' >> $HOME/.zshrc
