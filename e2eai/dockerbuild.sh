sudo buildkitd 
sudo docker build --no-cache -t automlpipeline:latest .
sudo docker tag automlpipeline:latest ppalmes/automlpipeline:latest
#sudo docker run -it --rm ppalmes/automlpipeline:latest julia --project main.jl 5 1
sudo docker login registry.docker.com
sudo docker push ppalmes/automlpipeline:latest


sudo docker build --no-cache -t amlp:latest -f ./Dockerfile.sysimage .
sudo docker tag amlp:latest ppalmes/amlp:latest
#sudo docker run -it --rm ppalmes/amlp:latest julia --project main.jl 5 1
sudo docker login registry.docker.com
sudo docker push ppalmes/amlp:latest
