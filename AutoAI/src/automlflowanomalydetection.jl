module AutoMLFlowAnomalyDetections
using Statistics
using Serialization
import PythonCall
const PYC = PythonCall

using DataFrames: DataFrame
using Random
using ..AbsTypes
using ..Utils
using ..AutoAnomalyDetections
using ..AutoMLPipeline: getiris

import ..AbsTypes: fit, fit!, transform, transform!
export fit, fit!, transform, transform!
export mlfaddriver, AutoMLFlowAnomalyDetection

const MLF = PYC.pynew()
const REQ = PYC.pynew()

function __init__()
  PYC.pycopy!(MLF, PYC.pyimport("mlflow"))
  PYC.pycopy!(REQ, PYC.pyimport("requests"))
end

include("./mlflowutils.jl")

mutable struct AutoMLFlowAnomalyDetection <: Workflow
  name::String
  model::Dict{Symbol,Any}

  function AutoMLFlowAnomalyDetection(args=Dict())
    default_args = Dict(
      :name => "AutoAnomalDetection",
      :projectname => "AutoAnomalDetection",
      :url => "http://localhost:8080",
      :description => "Automated Anomaly Detection",
      :projecttype => "anomalydetection",
      :artifact_name => "autoad.bin",
      :votepercent => 0.0,
      :impl_args => Dict()
    )
    cargs = nested_dict_merge(default_args, args)
    initmlflowcargs!(cargs)
    new(cargs[:name], cargs)
  end
end

function AutoMLFlowAnomalyDetection(name::String, args::Dict)
  AutoMLFlowAnomalyDetection(Dict(:name => name, args...))
end

function AutoMLFlowAnomalyDetection(name::String; args...)
  AutoMLFlowAnomalyDetection(Dict(Dict(pairs(args))...))
end

function (obj::AutoMLFlowAnomalyDetection)(; args...)
  model = obj.model
  cargs = nested_dict_merge(model, Dict(pairs(args)))
  obj.model = cargs
  return obj
end

function fit!(mlfad::AutoMLFlowAnomalyDetection, X::DataFrame, Y::Vector)
  setupautofit!(mlfad)
  # automate anomaly detection
  votepercent = mlfad.model[:votepercent]
  autoad = AutoAnomalyDetection(Dict(:votepercent => votepercent))
  adoutput = fit_transform!(autoad, X, Y)
  # save model in memory
  mlfad.model[:automodel] = autoad
  # log info to mlflow
  MLF.log_param("ADOutput", adoutput)
  MLF.log_metric("votepercent", autoad.model[:votepercent])
  # log artifacts, end experiment run
  logmlartifact(mlfad)
end

function fit(mlfad::AutoMLFlowAnomalyDetection, X::DataFrame, Y::Vector)
  mlfcopy = deepcopy(mlfad)
  fit!(mlfcopy, X, Y)
  return mlfcopy
end

function transform!(mlfad::AutoMLFlowAnomalyDetection, X::DataFrame)
  return autotransform!(mlfad, X)
end

function transform(mlfad::AutoMLFlowAnomalyDetection, X::DataFrame)
  mlfadc = deepcopy(mlfad)
  return transform!(mlfadc, X)
end

function mlfaddriver()
  url = "http://mlflow.home"

  X = vcat(5 * cos.(-10:10), sin.(-30:30), 3 * cos.(-10:10), 2 * tan.(-10:10), sin.(-30:30)) |> x -> DataFrame([x], :auto)

  mlfad = AutoMLFlowAnomalyDetection(Dict(:url => url))
  Yc = fit_transform!(mlfad, X)
  println(Yc |> x -> first(x, 5))

  # test prediction using exisiting trained model from artifacts
  run_id = mlfad.model[:run_id]
  newmlad = AutoMLFlowAnomalyDetection(Dict(:run_id => run_id, :url => url))
  newmlad = AutoMLFlowAnomalyDetection(Dict(:url => url))
  newmlad(; run_id, url)
  Yn = transform!(newmlad, X)
  println(Yc |> x -> first(x, 5))

  mlvad = AutoMLFlowAnomalyDetection(Dict(:url => url, :votepercent => 0.5))
  Yc = fit_transform!(mlvad, X)
  println(Yc |> x -> first(x, 5))

  return nothing
end

end
