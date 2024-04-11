using PackageCompiler
create_sysimage(["AMLPipelineBase","DataFrames","StatsBase",
                  "CSV","Dates","Distributed","TSML",
                  "Random","ArgParse","Test",
                  "Statistics","Serialization"], 
                  sysimage_path="amlp.so", precompile_execution_file="precompile.jl")

