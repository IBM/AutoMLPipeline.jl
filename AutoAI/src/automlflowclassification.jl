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
      :name => "AutoMLClassifications",
      :projectname => "AutoMLClassifications",
      :url => "http://localhost:8080",
      :description => "Automated Classification",
      :projecttype => "classification",
      :artifact_name => "AutoClassificationModel.bin",
      :impl_args => Dict(
        :name => "autoclass",
        :complexity => "low",
        :prediction_type => "classification",
        :nfolds => 3,
        :metric => "balanced_accuracy_score",
        :nworkers => 5,
        :learners => ["rfc", "rbfsvc", "gbc", "adac"],
        :scalers => ["norm", "pt", "mx", "std", "rb", "pt", "noop"],
        :extractors => ["pca", "ica", "fa", "noop"],
        :sortrev => true
      )
    )
    cargs = nested_dict_merge(default_args, args)
    initmlflowcargs!(cargs)
    cargs[:automodel] = AutoClassification(cargs[:impl_args])
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
  r(x) = round(x, digits=2)
  # start experiment run
  setupautofit!(mlfcl)
  # automate classification
  autoclass = mlfcl.model[:automodel]
  fit_transform!(autoclass, X, Y)
  # save model in memory
  mlfcl.model[:automodel] = autoclass
  # log info to mlflow
  bestmodel = autoclass.model[:bestpipeline].model[:description]
  MLF.log_param("best_pipeline", bestmodel)
  MLF.log_param("searched_pipelines", autoclass.model[:dfpipelines].Description)
  bestmean = autoclass.model[:performance].mean[1]
  bestsd = autoclass.model[:performance].sd[1]
  MLF.log_metric("best_pipeline_mean", r(bestmean))
  MLF.log_metric("best_pipeline_sd", r(bestsd))
  # log artifacts, end experiment run
  logmlartifact(mlfcl)
  @info "saved model runid: $(mlfcl.model[:run_id])"
  @info "saved model uri: $(mlfcl.model[:bestmodel_uri])"
end

function fit(mlfcl::AutoMLFlowClassification, X::DataFrame, Y::Vector)
  mlfcopy = deepcopy(mlfcl)
  fit!(mlfcopy, X, Y)
  return mlfcopy
end

function transform!(mlfcl::AutoMLFlowClassification, X::DataFrame)
  # start experiment run
  Y = autotransform!(mlfcl, X)
  # end run
  MLF.end_run()
  return Y
end

function mlfcldriver()
  url = "http://mlflow.home"
  url = "http://localhost:8080"

  df = getiris()
  X = df[:, 1:end-1]
  Y = df[:, end] |> collect

  mlfclass = AutoMLFlowClassification(Dict(:url => url))
  Yc = fit_transform!(mlfclass, X, Y)
  println("accuracy = ", mean(Y .== Yc))

  newmfclass = AutoMLFlowClassification(Dict(:url => url, :impl_args => Dict(:nfolds => 2)))
  Yc = fit_transform!(newmfclass, X, Y)
  println("accuracy = ", mean(Y .== Yc))

  nclass = AutoMLFlowClassification(Dict(:url => url))
  nclass.model[:automodel](; nfolds=2)
  Yc = fit_transform!(nclass, X, Y)
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
