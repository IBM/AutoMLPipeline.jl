module MyMain
using Distributed

workers=5
nprocs() == 1 && addprocs(workers;exeflags=["--project=$(Base.active_project())"])

@everywhere using ArgParse
@everywhere using DataFrames
@everywhere using CSV


# disable warnings
@everywhere import PythonCall
@everywhere const PYC=PythonCall
@everywhere warnings = PYC.pyimport("warnings")
@everywhere warnings.filterwarnings("ignore")

include("twoblocks.jl")
using .TwoBlocksPipeline
@everywhere include("twoblocks.jl")
@everywhere using .TwoBlocksPipeline

@everywhere function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--preprocessing_level", "-l"
            help = "preprocessing level"
            arg_type = String
            default = "low"
        "--pipeline_complexity", "-c"
            help = "pipeline complexity"
            arg_type = String
            default = "low"
        "--no-save"
            help = "save model"
            action = :store_true
        "input_csvfile"
            help = "input csv file"
            required = true
    end

    return parse_args(s)
end

@everywhere function mymain()
    parsed_args = parse_commandline()
    println("Parsed args:")
    for (arg,val) in parsed_args
        println("  $arg  =>  $val")
    end
    fname = parsed_args["input_csvfile"]
    data = CSV.read(fname,DataFrame)
    X = data[:,1:(end-1)]
    Y = data[:,end] |> collect
    twoblockspipelinesearch(X,Y)
end
end

mymain()
