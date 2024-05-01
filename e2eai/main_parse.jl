using Distributed
using ArgParse
using CSV
using DataFrames

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--pipeline_complexity", "-c"
            help = "pipeline complexity"
            arg_type = String
            default = "low"
        "--prediction_type", "-t"
            help = "classification or regression"
            arg_type = String
            default = "classification"
        "--nfolds", "-f"
            help = "number of crossvalidation folds"
            arg_type = Int64
            default = 3
        "--workers", "-w"
            help = "number of workers"
            arg_type = Int64
            default = 5
        "--no_save"
            help = "save model"
            action = :store_true
        "csvfile"
            help = "input csv file"
            required = true
            #default="iris.csv"
    end
    return parse_args(s;as_symbols=true)
end

#function extract_args()
#    parsed_args = parse_commandline()
#    println("Parsed args:")
#    for (arg,val) in parsed_args
#        println("  $arg  =>  $val")
#    end
#    fname = parsed_args[:input_csvfile]
#    data = CSV.read(fname,DataFrame)
#    X = data[:,1:(end-1)]
#    Y = data[:,end] |> collect
#    #return(workers,X,Y)
#    return (;parsed_args...)
#end
#
#const (csv,X,Y)=extract_args()
const _cliargs = (;parse_commandline()...)
const _workers = _cliargs[:workers]

nprocs() == 1 && addprocs(_workers;exeflags=["--project=$(Base.active_project())"])

if _cliargs[:prediction_type] == "classification"
    @everywhere include("pipelineblocksclassification.jl")
    @everywhere using .PipelineBlocksClassification:twoblockspipelinesearch
    @everywhere using .PipelineBlocksClassification:oneblockpipelinesearch
elseif _cliargs[:prediction_type] == "regression"
    @everywhere include("pipelineblocksregression.jl")
    @everywhere using .PipelineBlocksRegression:twoblockspipelinesearch
    @everywhere using .PipelineBlocksRegression:oneblockpipelinesearch
else
    error("cli argument error for prediction type")
end

function mymain()
    fname = _cliargs[:csvfile]
    data = CSV.read(fname,DataFrame)
    X = data[:,1:(end-1)]
    Y = data[:,end] |> collect
    best = if _cliargs[:pipeline_complexity] == "low"
        oneblockpipelinesearch(X,Y;nfolds=_cliargs[:nfolds])
    else
        twoblockspipelinesearch(X,Y;nfolds=_cliargs[:nfolds])
    end
    r(x)=round(x,digits=2)
    println("best model: ",best[1])
    println(" mean ± sd: ",r(best[2])," ± ",r(best[3]))
    return best
end
mymain()
