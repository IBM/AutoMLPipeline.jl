using Distributed
using ArgParse
using CSV
using DataFrames: DataFrame
using AutoAI
using Statistics


function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--url", "-u"
        help = "mlflow server url"
        arg_type = String
        default = "http://localhost:8080"
        "--prediction_type", "-t"
        help = "classification, regression, anomalydetection"
        arg_type = String
        default = "classification"
        "--complexity", "-c"
        help = "pipeline complexity"
        arg_type = String
        default = "low"
        "--output_file", "-o"
        help = "output location"
        arg_type = String
        default = "NONE"
        "--nfolds", "-f"
        help = "number of crossvalidation folds"
        arg_type = Int64
        default = 3
        "--nworkers", "-w"
        help = "number of workers"
        arg_type = Int64
        default = 5
        "--no_save"
        help = "save model"
        action = :store_true
        "--predict_only", "-p"
        help = "no training, predict only"
        action = :store_true
        "--runid", "-r"
        help = "runid of experiment for trained model"
        arg_type = String
        default = "NONE"
        "csvfile"
        help = "input csv file"
        required = true
    end
    return parse_args(s; as_symbols=true)
end

const _cliargs = parse_commandline()
const _workers = _cliargs[:nworkers]

if _cliargs[:predict_only] == false
    nprocs() == 1 && addprocs(_workers; exeflags=["--project=$(Base.active_project())"])
    @everywhere using AutoAI
end

function autoclassmode(args::Dict)
    url = args[:url]
    complexity = args[:complexity]
    nfolds = args[:nfolds]
    nworkers = args[:nworkers]
    prediction_type = args[:prediction_type]
    impl_args = (; complexity, nfolds, nworkers, prediction_type) |> pairs |> Dict
    fname = _cliargs[:csvfile]
    df = CSV.read(fname, DataFrame)
    X = df[:, 1:end-1]
    Y = df[:, end] |> collect
    autoclass = AutoMLFlowClassification(Dict(:url => url, :impl_args => impl_args))
    Yc = fit_transform!(autoclass, X, Y)
    println("accuracy = ", mean(Y .== Yc))
    return autoclass
end

function autoregmode(args::Dict)
    url = args[:url]
    complexity = args[:complexity]
    nfolds = args[:nfolds]
    nworkers = args[:nworkers]
    prediction_type = args[:prediction_type]
    impl_args = (; complexity, nfolds, nworkers, prediction_type) |> pairs |> Dict
    fname = _cliargs[:csvfile]
    df = CSV.read(fname, DataFrame)
    X = df[:, 1:end-1]
    Y = df[:, end] |> collect
    autoreg = AutoMLFlowRegression(Dict(:url => url, :impl_args => impl_args))
    Yc = fit_transform!(autoreg, X, Y)
    println("mse = ", mean((Y - Yc) .^ 2))
    return autoreg
end

function doprediction_only(args::Dict)
    fname = args[:csvfile]
    X = CSV.read(fname, DataFrame)
    run_id = args[:runid]
    url = args[:url]
    mlf =
        predtype = args[:prediction_type]
    mlf = if predtype == "classification"
        AutoMLFlowClassification(Dict(:run_id => run_id, :url => url))
    elseif predtype == "regression"
        AutoMLFlowRegression(Dict(:run_id => run_id, :url => url))
    else
        error("unknown predtype option")
    end
    Yn = transform!(mlf, X)
    ofile = args[:output_file]
    if ofile != "NONE"
        open(ofile, "w") do stfile
            println(stfile, "prediction: $Yn")
            println(stdout, "prediction: $Yn")
        end
    else
        println(stdout, "prediction: $Yn")
    end
    return Yn
end

function printsummary(io::IO, automl::Workflow)
    r(x) = round(x, digits=2)
    trainedmodel = automl.model[:automodel]
    bestmodel = trainedmodel.model[:bestpipeline].model[:description]
    println(io, "pipelines: $(trainedmodel.model[:dfpipelines].Description)")
    println(io, "best_pipeline: $bestmodel")
    bestmean = trainedmodel.model[:performance].mean[1]
    bestsd = trainedmodel.model[:performance].sd[1]
    println(io, "best_pipeline_performance: $(r(bestmean)) ± $(r(bestsd))")
end

function dotrainandpredict(args::Dict)
    # train model
    predtype = args[:prediction_type]
    automl = if predtype == "classification"
        autoclassmode(args)
    elseif predtype == "regression"
        autoregmode(args)
    end
    ofile = args[:output_file]
    if ofile != "NONE"
        open(ofile, "w") do stfile
            printsummary(stfile, automl)
            printsummary(stdout, automl)
        end
    else
        printsummary(stdout, automl)
    end
end

function main(args::Dict)
    if args[:predict_only] == true
        # predict only using run_id of model in the artifact
        doprediction_only(args)
    else
        # train and predict
        dotrainandpredict(args)
    end
end
main(_cliargs)
