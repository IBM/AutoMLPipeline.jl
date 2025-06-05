import SimpleContainerGenerator

pkgs = ["ArgParse","AutoMLPipeline","CSV","DataFrames","Distributed","Random"]
julia_version = "1.10.2"

SimpleContainerGenerator.create_dockerfile(pkgs;
                                           output_directory=pwd(),
                                           julia_version = julia_version)

run(`nerdctl build -t ppalmes/automlpipeline .`)
