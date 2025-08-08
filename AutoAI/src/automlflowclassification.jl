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
  # end any running experiment
  # MLF.end_run()
  # generate run name
  run_name = mlfcl.model[:name] * "_" * "fit" * "_" * randstring(3)
  mlfcl.model[:run_name] = run_name
  MLF.set_experiment(mlfcl.model[:name])
  MLF.start_run(run_name=run_name)
  # get run_id
  run = MLF.active_run()
  mlfcl.model[:run_id] = run.info.run_id
  # automate classification
  autoclass = AutoClassification()
  fit_transform!(autoclass, X, Y)
  bestmodel = autoclass.model[:bestpipeline].model[:description]
  MLF.log_param("bestmodel", bestmodel)
  MLF.log_param("pipelines", autoclass.model[:dfpipelines].Description)
  MLF.log_metric("bestperformance", autoclass.model[:performance].mean[1])
  # save model in mlflow
  artifact_name = mlfcl.model[:artifact_name]
  serialize(artifact_name, autoclass)
  MLF.log_artifact(artifact_name)
  # save model in memory
  mlfcl.model[:autoclass] = autoclass
  bestmodel_uri = MLF.get_artifact_uri(artifact_path=artifact_name)
  # save model  uri location
  mlfcl.model[:bestmodel_uri] = bestmodel_uri
  MLF.end_run()
end

function fit(mlfcl::AutoMLFlowClassification, X::DataFrame, Y::Vector)
  mlfcopy = deepcopy(mlfcl)
  fit!(mlfcopy, X, Y)
  return mlfcopy
end

function transform!(mlfcl::AutoMLFlowClassification, X::DataFrame)
  MLF.end_run()
  # download model artifact
  run_id = mlfcl.model[:run_id]
  artifact_name = mlfcl.model[:artifact_name]

  try
    model_artifacts = MLF.artifacts.list_artifacts(run_id=run_id)
  catch e
    @info e
    throw("Artifact $artifact_name does not exist in run_id = $run_id")
  end

  run_name = mlfcl.model[:name] * "_" * "transform" * "_" * randstring(3)
  mlfcl.model[:run_name] = run_name
  MLF.set_experiment(mlfcl.model[:name])
  MLF.start_run(run_name=run_name)
  pylocalpath = MLF.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_name)
  bestmodel = deserialize(string(pylocalpath))
  Y = transform!(bestmodel, X)
  MLF.log_param("output", Y)
  MLF.end_run()
  return Y
end

function mlfcldriver()
  df = getiris()
  X = df[:, 1:end-1]
  Y = df[:, end] |> collect

  mlfclass = AutoMLFlowClassification()
  Yc = fit_transform!(mlfclass, X, Y)
  println("accuracy = ", mean(Y .== Yc))

  # test prediction using exisiting trained model from artifacts
  run_id = mlfclass.model[:run_id]
  newmfclass = AutoMLFlowClassification(Dict(:run_id => run_id))
  newmfclass = AutoMLFlowClassification()
  newmfclass(; run_id=run_id)
  Yn = transform!(newmfclass, X)
  println("accuracy = ", mean(Yn .== Y))

  return nothing
end

end
