From julia
MAINTAINER Paulito Palmes "paulpalmes@ie.ibm.com"
RUN mkdir /amlp
WORKDIR /amlp
COPY Project.toml /amlp/
COPY Manifest.toml /amlp/
RUN julia --project -e 'using Pkg;Pkg.instantiate();Pkg.update();Pkg.precompile()'
COPY * /amlp/
