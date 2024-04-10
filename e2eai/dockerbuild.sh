sudo docker build --no-cache -t automlpipeline:latest .
sudo docker tag automlpipeline:latest ppalmes/automlpipeline:latest
#sudo docker run -it --rm ppalmes/automlpipeline:latest julia --project main.jl 5 1
sudo docker login registry.docker.com
sudo docker push ppalmes/automlpipeline:latest
