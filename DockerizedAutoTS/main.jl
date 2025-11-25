using AutoAD
using ArgParse
using CSV
using DataFrames
using Statistics
using AutoAD


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
    default = "anomalydetection"
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

const _cliargs = parse_commandline()

function autoadmode(args::Dict)
  url = args[:url]
  votepercent = args[:votepercent]
  fname = args[:csvfile]
  df = CSV.read(fname, DataFrame)
  X = df[:, 1:1]
  autoad = AutoMLFlowAnomalyDetection(Dict(:url => url, :impl_args=>Dict(:votepercent=>votepercent)))
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
    autotsmode(args)
else
    @error "check cli arguments: $args"
  end
end

main(_cliargs)
