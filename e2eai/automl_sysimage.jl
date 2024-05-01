using PackageCompiler
create_sysimage(["AMLPipelineBase","DataFrames","StatsBase",
                  "CSV","Dates","Distributed",
                  "Random","ArgParse","Test",
                  "Statistics","Serialization"], 
                  sysimage_path="automl.so", precompile_execution_file="automl_precompile.jl")
