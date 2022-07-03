sudo docker build -t automlpipeline-arm:latest .
sudo docker tag automlpipeline-arm:latest ppalmes/automlpipeline-arm:latest
sudo docker login
sudo docker push ppalmes/automlpipeline-arm:latest
