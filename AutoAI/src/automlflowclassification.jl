module AutoMLFlowClassifications
using Statistics
using Serialization
import PythonCall
const PYC = PythonCall

using DataFrames: DataFrame
using Random
using ..AbsTypes
using ..Utils
using ..AutoClassifications
using ..AutoMLPipeline: getiris

import ..AbsTypes: fit, fit!, transform, transform!
export fit, fit!, transform, transform!
export mlfcldriver, AutoMLFlowClassification

const MLF = PYC.pynew()
const REQ = PYC.pynew()

function __init__()
  PYC.pycopy!(MLF, PYC.pyimport("mlflow"))
  PYC.pycopy!(REQ, PYC.pyimport("requests"))
end

include("./mlflowutils.jl")

mutable struct AutoMLFlowClassification <: Workflow
  name::String
  model::Dict{Symbol,Any}

  function AutoMLFlowClassification(args=Dict())
    default_args = Dict(
      :name => "AutoMLClassification",
      :projectname => "AutoMLClassification",
      :url => "http://localhost:8080",
      :description => "Automated Classification",
      :projecttype => "classification",
      :artifact_name => "autoclass.bin",
      :impl_args => Dict()
    )
    cargs = nested_dict_merge(default_args, args)
    initmlflowcargs!(cargs)
    new(cargs[:name], cargs)
  end
end

function AutoMLFlowClassification(name::String, args::Dict)
  AutoMLFlowClassification(Dict(:name => name, args...))
end

function AutoMLFlowClassification(name::String; args...)
  AutoMLFlowClassification(Dict(Dict(pairs(args))...))
end

function (obj::AutoMLFlowClassification)(; args...)
  model = obj.model
  cargs = nested_dict_merge(model, Dict(pairs(args)))
  obj.model = cargs
  return obj
end

function fit!(mlfcl::AutoMLFlowClassification, X::DataFrame, Y::Vector)
  # start experiment run
  setupautofit!(mlfcl)
  # automate classification
  autoclass = AutoClassification()
  fit_transform!(autoclass, X, Y)
  # save model in memory
  mlfcl.model[:automodel] = autoclass
  # log info to mlflow
  bestmodel = autoclass.model[:bestpipeline].model[:description]
  MLF.log_param("bestmodel", bestmodel)
  MLF.log_param("pipelines", autoclass.model[:dfpipelines].Description)
  MLF.log_metric("bestperformance", autoclass.model[:performance].mean[1])
  # log artifacts, end experiment run
  logmlartifact(mlfcl)
end

function fit(mlfcl::AutoMLFlowClassification, X::DataFrame, Y::Vector)
  mlfcopy = deepcopy(mlfcl)
  fit!(mlfcopy, X, Y)
  return mlfcopy
end

function transform!(mlfcl::AutoMLFlowClassification, X::DataFrame)
  return autotransform!(mlfcl, X)
end

function mlfcldriver()
  url = "http://mlflow.home"
  df = getiris()
  X = df[:, 1:end-1]
  Y = df[:, end] |> collect

  mlfclass = AutoMLFlowClassification(Dict(:url => url))
  Yc = fit_transform!(mlfclass, X, Y)
  println("accuracy = ", mean(Y .== Yc))

  # test prediction using exisiting trained model from artifacts
  run_id = mlfclass.model[:run_id]
  newmfclass = AutoMLFlowClassification(Dict(:run_id => run_id, :url => url))
  newmfclass = AutoMLFlowClassification(Dict(:url => url))
  newmfclass(; run_id=run_id)
  Yn = transform!(newmfclass, X)
  println("accuracy = ", mean(Yn .== Y))

  return nothing
end

end
