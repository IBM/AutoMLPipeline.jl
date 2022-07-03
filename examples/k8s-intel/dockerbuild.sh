sudo docker build -t automlpipeline-intel:latest .
sudo docker tag automlpipeline-intel:latest ppalmes/automlpipeline-intel:latest
sudo docker login
sudo docker push ppalmes/automlpipeline-intel:latest

