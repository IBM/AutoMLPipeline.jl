using Distributed
using ArgParse
using CSV
using DataFrames

function parse_commandline()
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
        "--workers", "-w"
            help = "number of workers"
            arg_type = Integer
            default = 5
        "--no-save"
            help = "save model"
            action = :store_true
        "input_csvfile"
            help = "input csv file"
            required = true
    end
    return parse_args(s)
end

function extract_args()
    parsed_args = parse_commandline()
    println("Parsed args:")
    for (arg,val) in parsed_args
        println("  $arg  =>  $val")
    end
    workers=parsed_args["workers"]
    fname = parsed_args["input_csvfile"]
    data = CSV.read(fname,DataFrame)
    X = data[:,1:(end-1)]
    Y = data[:,end] |> collect
    return(workers,X,Y)
end

const (workers,X,Y)=extract_args()

nprocs() == 1 && addprocs(workers;exeflags=["--project=$(Base.active_project())"])

@everywhere include("twoblocks.jl")
@everywhere using .TwoBlocksPipeline:twoblockspipelinesearch

twoblockspipelinesearch(X,Y)
