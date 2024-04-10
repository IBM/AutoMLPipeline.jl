using PackageCompiler

create_sysimage(["AutoMLPipeline"], sysimage_path="AutoML.so", precompile_execution_file="precompile.jl")

