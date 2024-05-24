module End2EndAI
__precompile__(false)
using DataFrames
using CSV
using ArgParse
using Distributed

export automlmain

workers=5
nprocs() == 1 && addprocs(workers;  exeflags=["--project=$(Base.active_project())"])

include("twoblocks.jl")
@everywhere include("twoblocks.jl")
@everywhere using .TwoBlocksPipeline

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--pipeline_level"
            help = "small, medium, large"
            arg_type = String
            default = "small"
        "--model"
            help = "small, medium, large"
            arg_type = String
            default = "small"
        "--save"
            help = "save results"
            action = :store_true
        "csvfilename"
            help = "csv file to process"
            default = "iris.csv"
            #required = true
    end

    return parse_args(s,as_symbols=true)
end

function automlmain()
    parsed_args = parse_commandline()
    createrunpipeline(parsed_args)
end

function createrunpipeline(args::Dict)
    fname = args[:csvfilename]
    df = CSV.read(fname,DataFrame)
    X = df[:,1:(end-1)]
    Y = df[:,end] |> collect
    results = TwoBlocksPipeline.twoblockspipelinesearch(X,Y)
    println(results[1:1,:])
end

end
