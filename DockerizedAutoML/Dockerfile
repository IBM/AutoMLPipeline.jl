FROM julia:latest
LABEL MAINTAINER='Paulito Palmes "paulpalmes@ie.ibm.com"'
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y locales build-essential git 

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN mkdir /amlp
RUN cd /amlp
WORKDIR /amlp
COPY Project.toml /amlp/
COPY *.csv /amlp/
ADD https://github.com/IBM/AutoMLPipeline.jl.git /amlp/AutoMLPipeline/
RUN julia --project -e 'using Pkg;\
  Pkg.develop(path="./AutoMLPipeline/AutoAI/");\
  Pkg.instantiate();\
  Pkg.update();\
  Pkg.precompile();\
  using AutoAI;'
COPY main.jl /amlp/
ENTRYPOINT ["julia", "--project", "--", "main.jl"]
#ENTRYPOINT [ "/bin/bash" ]
