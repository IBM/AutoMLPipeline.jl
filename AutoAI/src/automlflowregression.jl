module AutoMLFlowRegressions
using Statistics
using Serialization
import PythonCall
const PYC = PythonCall

using DataFrames: DataFrame
using Random
using ..AbsTypes
using ..Utils
using ..AutoRegressions
using ..AutoMLPipeline: getiris

import ..AbsTypes: fit, fit!, transform, transform!
export fit, fit!, transform, transform!
export mlfregdriver, AutoMLFlowRegression

const MLF = PYC.pynew()
const REQ = PYC.pynew()

function __init__()
  PYC.pycopy!(MLF, PYC.pyimport("mlflow"))
  PYC.pycopy!(REQ, PYC.pyimport("requests"))
end

include("./mlflowutils.jl")

mutable struct AutoMLFlowRegression <: Workflow
  name::String
  model::Dict{Symbol,Any}

  function AutoMLFlowRegression(args=Dict())
    default_args = Dict(
      :name => "AutoMLRegressions",
      :projectname => "AutoMLRegressions",
      :url => "http://localhost:8080",
      :description => "Automated Regression",
      :projecttype => "regression",
      :artifact_name => "AutoRegressionModel.bin",
      :impl_args => Dict(
        :name => "autoreg",
        :complexity => "low",
        :prediction_type => "regression",
        :nfolds => 3,
        :metric => "mean_squared_error",
        :nworkers => 5,
        :learners => ["rfr", "svr", "gbr", "adar"],
        :scalers => ["norm", "pt", "mx", "std", "rb", "pt", "noop"],
        :extractors => ["pca", "ica", "fa", "noop"],
        :sortrev => false
      )
    )
    cargs = nested_dict_merge(default_args, args)
    initmlflowcargs!(cargs)
    cargs[:automodel] = AutoRegression(cargs[:impl_args])
    new(cargs[:name], cargs)
  end
end

function AutoMLFlowRegression(name::String, args::Dict)
  AutoMLFlowRegression(Dict(:name => name, args...))
end

function AutoMLFlowRegression(name::String; args...)
  AutoMLFlowRegression(Dict(Dict(pairs(args))...))
end

function (obj::AutoMLFlowRegression)(; args...)
  model = obj.model
  cargs = nested_dict_merge(model, Dict(pairs(args)))
  obj.model = cargs
  return obj
end

function fit!(mlfreg::AutoMLFlowRegression, X::DataFrame, Y::Vector)
  r(x) = round(x, digits=2)
  # start experiment run
  setupautofit!(mlfreg)
  # automate regression
  autoreg = mlfreg.model[:automodel]
  fit_transform!(autoreg, X, Y)
  # save model in memory
  mlfreg.model[:automodel] = autoreg
  # log info to mlflow
  bestmodel = autoreg.model[:bestpipeline].model[:description]
  MLF.log_param("best_pipeline", bestmodel)
  MLF.log_param("searched_pipelines", autoreg.model[:dfpipelines].Description)
  bestmean = autoreg.model[:performance].mean[1]
  bestsd = autoreg.model[:performance].sd[1]
  MLF.log_metric("best_pipeline_mean", r(bestmean))
  MLF.log_metric("best_pipeline_sd", r(bestsd))
  # log artifacts, end experiment run
  logmlartifact(mlfreg)
end

function fit(mlfreg::AutoMLFlowRegression, X::DataFrame, Y::Vector)
  mlfcopy = deepcopy(mlfreg)
  fit!(mlfcopy, X, Y)
  return mlfcopy
end

function transform!(mlfreg::AutoMLFlowRegression, X::DataFrame)
  # start experiment run
  Y = autotransform!(mlfreg, X)
  # end run
  MLF.end_run()
  return Y
end

function mlfregdriver()
  url = "http://mlflow.home"
  url = "http://localhost:8080"

  df = getiris()
  X = df[:, [1, 2, 3, 5]]
  Y = df[:, 4] |> collect

  #mlfreg = AutoMLFlowRegression(Dict(:url => url))
  #Yc = fit_transform!(mlfreg, X, Y)
  #println("mse = ", mean((Y - Yc) .^ 2))

  #newmfreg = AutoMLFlowRegression(Dict(:url => url, :impl_args => Dict(:nfolds => 2)))
  #Yn = fit_transform!(newmfreg, X,Y)
  #println("mse = ", mean((Y - Yn) .^ 2))

  newmfreg = AutoMLFlowRegression(Dict(:url => url))
  newmfreg.model[:automodel](; nfolds=5)
  Yn = fit_transform!(newmfreg, X, Y)
  println("mse = ", mean((Y - Yn) .^ 2))

  ### test prediction using exisiting trained model from artifacts
  #run_id = mlfreg.model[:run_id]
  ##run_id = "d7ea4d0582bb4519a96b36efbe1eda6a"
  #newmfreg = AutoMLFlowRegression(Dict(:run_id => run_id, :url => url))
  #newmfreg = AutoMLFlowRegression(Dict(:url => url))
  #newmfreg(; run_id, url)
  #Yn = transform!(newmfreg, X)
  #println("mse = ", mean((Y - Yn) .^ 2))

  return nothing
end

end
