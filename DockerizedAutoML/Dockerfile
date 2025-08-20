FROM julia:latest
LABEL MAINTAINER='Paulito Palmes "paulpalmes@ie.ibm.com"'
RUN apt-get update && apt-get install build-essential -y
RUN mkdir /amlp
RUN cd /amlp
WORKDIR /amlp
COPY Project.toml /amlp/
COPY *.csv /amlp/
ADD https://github.com/IBM/AutoMLPipeline.jl.git /amlp/AutoMLPipeline/
RUN julia --project -e 'using Pkg;\
  Pkg.develop(path="./AutoMLPipeline/AutoAI/");\
  Pkg.instantiate();\
  Pkg.update()'
COPY main.jl /amlp/
#ENTRYPOINT ["julia", "--project", "--", "main.jl"]
ENTRYPOINT [ "/bin/bash" ]
