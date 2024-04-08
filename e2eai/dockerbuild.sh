nerdctl build -t automlpipeline:latest .
nerdctl tag automlpipeline:latest ppalmes/automlpipeline:latest
nerdctl run -it --rm automlpipeline:latest julia --project main.jl 5 1
nerdctl login registry.docker.com
nerdctl push ppalmes/automlpipeline:latest

