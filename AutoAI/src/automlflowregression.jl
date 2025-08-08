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

mutable struct AutoMLFlowRegression <: Workflow
  name::String
  model::Dict{Symbol,Any}

  function AutoMLFlowRegression(args=Dict())
    default_args = Dict(
      :name => "AutoMLRegression",
      :projectname => "AutoMLRegression",
      :url => "http://localhost:8080",
      :description => "Automated Regression",
      :projecttype => "regression",
      :artifact_name => "autoreg.bin",
      :impl_args => Dict()
    )
    cargs = nested_dict_merge(default_args, args)
    #cargs[:name] = cargs[:name] * "_" * randstring(3)
    experiment_tags = Dict(
      "projectname" => cargs[:projectname],
      "projecttype" => cargs[:projecttype],
      "notes" => cargs[:description]
    )
    # check if mlflow server exists
    try
      httpget = getproperty(REQ, "get")
      res = httpget(cargs[:url] * "/health")
    catch
      @error("Mlflow Server Unreachable")
      exit(1)
    end
    MLF.set_tracking_uri(uri=cargs[:url])
    name = cargs[:name]
    experiment = MLF.search_experiments(filter_string="name = \'$name\'")
    if PYC.pylen(experiment) != 0
      MLF.set_experiment(experiment[0].name)
    else
      theexperiment = MLF.create_experiment(name=name, tags=experiment_tags)
      cargs[:experiment_id] = theexperiment
    end
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
  MLF.end_run()
  # end any running experiment
  # MLF.end_run()
  # generate run name
  run_name = mlfreg.model[:name] * "_" * "fit" * "_" * randstring(3)
  mlfreg.model[:run_name] = run_name
  MLF.set_experiment(mlfreg.model[:name])
  MLF.start_run(run_name=run_name)
  # get run_id
  run = MLF.active_run()
  mlfreg.model[:run_id] = run.info.run_id
  # automate regression
  autoreg = AutoRegression()
  fit_transform!(autoreg, X, Y)
  bestmodel = autoreg.model[:bestpipeline].model[:description]
  MLF.log_param("bestmodel", bestmodel)
  MLF.log_param("pipelines", autoreg.model[:dfpipelines].Description)
  MLF.log_metric("bestperformance", autoreg.model[:performance].mean[1])
  # save model in mlflow
  artifact_name = mlfreg.model[:artifact_name]
  serialize(artifact_name, autoreg)
  MLF.log_artifact(artifact_name)
  # save model in memory
  mlfreg.model[:autoreg] = autoreg
  bestmodel_uri = MLF.get_artifact_uri(artifact_path=artifact_name)
  # save model  uri location
  mlfreg.model[:bestmodel_uri] = bestmodel_uri
  MLF.end_run()
end

function fit(mlfreg::AutoMLFlowRegression, X::DataFrame, Y::Vector)
  mlfcopy = deepcopy(mlfreg)
  fit!(mlfcopy, X, Y)
  return mlfcopy
end

function transform!(mlfreg::AutoMLFlowRegression, X::DataFrame)
  MLF.end_run()
  # download model artifact
  run_id = mlfreg.model[:run_id]
  model_artifacts = MLF.artifacts.list_artifacts(run_id=run_id)
  if PYC.pylen(model_artifacts) == 0
    @error "Artifact does not exist in run_id = $run_id"
    exit(1)
  end
  run_name = mlfreg.model[:name] * "_" * "transform" * "_" * randstring(3)
  mlfreg.model[:run_name] = run_name
  MLF.set_experiment(mlfreg.model[:name])
  MLF.start_run(run_name=run_name)
  artifact_name = mlfreg.model[:artifact_name]
  pylocalpath = MLF.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_name)
  bestmodel = deserialize(string(pylocalpath))
  Y = transform!(bestmodel, X)
  MLF.log_param("output", Y)
  MLF.end_run()
  return Y
end

function mlfregdriver()
  df = getiris()
  X = df[:, [1, 2, 3, 5]]
  Y = df[:, 4] |> collect

  mlfreg = AutoMLFlowRegression()
  Yc = fit_transform!(mlfreg, X, Y)
  println("mse = ", mean((Y - Yc) .^ 2))

  ### test prediction using exisiting trained model from artifacts
  run_id = mlfreg.model[:run_id]
  newmfreg = AutoMLFlowRegression(Dict(:run_id => run_id))
  newmfreg = AutoMLFlowRegression()
  newmfreg(; run_id=run_id)
  Yn = transform!(newmfreg, X)
  println("mse = ", mean((Y - Yn) .^ 2))

  return nothing
end

end
