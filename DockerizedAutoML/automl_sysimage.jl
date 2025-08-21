using PackageCompiler
create_sysimage(["AMLPipelineBase", "DataFrames", "StatsBase",
    "ArgParse", "AutoMLPipeline", "CSV", "Dates", "Distributed",
    "Random", "ArgParse", "Test", "Distributed", "PythonCall",
    "Statistics", "Serialization", "StatsBase", "Test"],
  sysimage_path="automl.so", precompile_execution_file="automl_precompile.jl")
