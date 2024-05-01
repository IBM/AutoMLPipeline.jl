#From --platform=linux/amd64 julia:latest
# for nerdctl
From julia:latest
MAINTAINER Paulito Palmes "paulpalmes@ie.ibm.com"
RUN mkdir /amlp
RUN cd /amlp
WORKDIR /amlp
COPY Project.toml /amlp/
RUN julia --project -e 'using Pkg;\
       Pkg.instantiate();\
       Pkg.update();\
       Pkg.precompile();\
       include("automl_sysimage.jl")'
COPY * /amlp/
