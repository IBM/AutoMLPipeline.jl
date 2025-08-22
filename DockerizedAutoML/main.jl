using Distributed
using ArgParse
using CSV
using DataFrames
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
    "--predict_only"
    help = "no training, predict only"
    action = :store_true
    "--runid"
    help = "runid of experiment for trained model"
    arg_type = String
    default = "NONE"
    "csvfile"
    help = "input csv file"
    required = true
  end
  return parse_args(s; as_symbols=true)
end

const _cliargs = (; parse_commandline()...)
const _workers = _cliargs[:workers]

nprocs() == 1 && addprocs(_workers; exeflags=["--project=$(Base.active_project())"])

@everywhere using AutoAI

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
end

function main()
  predtype = _cliargs[:prediction_type]
  if predtype == "classification"
    autoclassmode(_cliargs)
  elseif predtype == "regression"
    autoregmode(_cliargs)
end
main()
