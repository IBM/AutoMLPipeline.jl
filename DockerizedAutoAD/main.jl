using AutoAD: AutoMLFlowAnomalyDetections
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
    help = "timeseriesprediction, anomalydetection"
    arg_type = String
    default = "anomalydetecion"
    "--output_file", "-o"
    help = "output location"
    arg_type = String
    default = "NONE"
    "--votepercent", "-v"
    help = "votepercent for anomalydetection ensembles"
    arg_type = Float64
    default = 0.0
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
#_cliargs = parse_commandline()
const _workers = _cliargs[:workers]
#_workers = _cliargs[:nworkers]

nprocs() == 1 && addprocs(_workers; exeflags=["--project=$(Base.active_project())"])

@everywhere using AutoAD

function autoadmode(args::Dict)
  url = args[:url]
  prediction_type = args[:prediction_type]
  votepercent = args[:votepercent]
  dictargs = (; votepercent, prediction_type) |> pairs |> Dict
  fname = args[:csvfile]
  df = CSV.read(fname, DataFrame)
  X = df[:, 1:1]
  autoad = AutoMLFlowAnomalyDetections(Dict(:url => url, dictargs...))
  Yc = fit_transform!(autoad, X)
  println("output:", Yc |> x -> first(x, 5))
  return Yc
end

function autotsmode(args::Dict)
end

function main(args)
  predtype = args[:prediction_type]
  if predtype == "anomalydetection"
    autoadmode(args)
  elseif predtype == "timeseriesprediction"
    autotsmode(_cliargs)
  elseif predtype == "anomalydetection"
    autoadmode(_cliargs)
  end
end

main(_cliargs)
