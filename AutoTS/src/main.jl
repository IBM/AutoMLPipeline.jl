using AutoTS
using ArgParse
using CSV
using DataFrames
using Statistics


function parse_commandline()
  s = ArgParseSettings()
  @add_arg_table! s begin
    "--url", "-u"
    help = "mlflow server url"
    arg_type = String
    default = "http://localhost:8081"
    "--output_file", "-o"
    help = "output location"
    arg_type = String
    default = "NONE"
    "--learner", "-l"
    help = "learner"
    arg_type = String
    default = "auto"
    "--forecast_horizon", "-f"
    help = "forecast horizon"
    arg_type = Int64
    default = 10
    "--runid", "-r"
    help = "runid of experiment for trained model"
    arg_type = String
    default = "NONE"
    "--predict_only", "-p"
    help = "no training, predict only"
    action = :store_true
    "csvfile"
    help = "input csv file"
    required = true
  end
  return parse_args(s; as_symbols=true)
end

function doprediction_only(args::Dict)
  fname = args[:csvfile]
  X = CSV.read(fname, DataFrame)
  run_id = args[:runid]
  url = args[:url]
  predtype = args[:prediction_type]
  mlf = AutoMLFlowTSPrediction((Dict(:rund_id => run_id, :url => url)))
  Yn = transform!(mlf, X)
  ofile = args[:output_file]
  if ofile != "NONE"
    open(ofile, "w") do stfile
      println(stfile, "prediction: $Yn")
    end
  end
  println(stdout, "prediction: $Yn")
  return Yn
end

function dotrainandpredict(args::Dict)
  url = args[:url]
  learner = args[:learner]
  forecast_horizon = args[:forecast_horizon]
  fname = args[:csvfile]
  df = CSV.read(fname, DataFrame)
  X = df[:, 1:1]
  autots = AutoMLFlowTSPrediction(Dict(:url => url, :impl_args => Dict(:forecast_horizon => forecast_horizon, :learner => learner)))
  Yc = fit_transform!(autots, X)
  println("output:", Yc |> x -> first(x, 5))
  return Yc
end

function @main(MyARGS)
  ARGS = parse_commandline()
  if ARGS[:predict_only] == true
    # predict only using run_id of model in the artifact
    doprediction_only(ARGS)
  else
    # train and predict
    dotrainandpredict(ARGS)
  end
end
