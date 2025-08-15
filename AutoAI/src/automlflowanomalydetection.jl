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
  # end any running experiment
  # MLF.end_run()
  # generate run name
  run_name = mlfad.model[:name] * "_" * "fit" * "_" * randstring(3)
  mlfad.model[:run_name] = run_name
  MLF.set_experiment(mlfad.model[:name])
  MLF.start_run(run_name=run_name)
  # get run_id
  run = MLF.active_run()
  mlfad.model[:run_id] = run.info.run_id
  # automate anomaly detection
  votepercent = mlfad.model[:votepercent]
  autoad = AutoAnomalyDetection(Dict(:votepercent => votepercent))
  adoutput = fit_transform!(autoad, X, Y)
  MLF.log_param("ADOutput", adoutput)
  MLF.log_metric("votepercent", autoad.model[:votepercent])
  # save model in mlflow
  artifact_name = mlfad.model[:artifact_name]
  # use temporary directory
  tmpdir = tempdir()
  artifact_location = joinpath(tmpdir, artifact_name)
  serialize(artifact_location, autoad)
  MLF.log_artifact(artifact_location)
  # save model in memory
  mlfad.model[:autoad] = autoad
  bestmodel_uri = MLF.get_artifact_uri(artifact_path=artifact_name)
  # save model  uri location
  mlfad.model[:bestmodel_uri] = bestmodel_uri
  MLF.end_run()
end

function fit(mlfad::AutoMLFlowAnomalyDetection, X::DataFrame, Y::Vector)
  mlfcopy = deepcopy(mlfad)
  fit!(mlfcopy, X, Y)
  return mlfcopy
end

function transform!(mlfad::AutoMLFlowAnomalyDetection, X::DataFrame)
  MLF.end_run()
  # download model artifact
  run_id = mlfad.model[:run_id]
  artifact_name = mlfad.model[:artifact_name]

  try
    model_artifacts = MLF.artifacts.list_artifacts(run_id=run_id)
    @assert model_artifacts[0].path |> string == "autoad.bin"
  catch e
    @info e
    throw("Artifact $artifact_name does not exist in run_id = $run_id")
  end

  run_name = mlfad.model[:name] * "_" * "transform" * "_" * randstring(3)
  mlfad.model[:run_name] = run_name
  MLF.set_experiment(mlfad.model[:name])
  MLF.start_run(run_name=run_name)
  pylocalpath = MLF.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_name)
  bestmodel = deserialize(string(pylocalpath))
  Y = transform!(bestmodel, X)
  MLF.log_param("output", Y)
  MLF.end_run()
  return Y
end

function transform(mlfad::AutoMLFlowAnomalyDetection, X::DataFrame)
  mlfadc = deepcopy(mlfad)
  return transform!(mlfadc, X)
end

function mlfaddriver()

  X = vcat(5 * cos.(-10:10), sin.(-30:30), 3 * cos.(-10:10), 2 * tan.(-10:10), sin.(-30:30)) |> x -> DataFrame([x], :auto)

  mlfad = AutoMLFlowAnomalyDetection()
  Yc = fit_transform!(mlfad, X)
  println(Yc |> x -> first(x, 5))

  # test prediction using exisiting trained model from artifacts
  run_id = mlfad.model[:run_id]
  newmlad = AutoMLFlowAnomalyDetection(Dict(:run_id => run_id))
  newmlad = AutoMLFlowAnomalyDetection()
  newmlad(; run_id=run_id)
  Yn = transform!(newmlad, X)
  println(Yc |> x -> first(x, 5))

  mlvad = AutoMLFlowAnomalyDetection(Dict(:votepercent => 0.5))
  Yc = fit_transform!(mlvad, X)
  println(Yc |> x -> first(x, 5))

  return nothing
end

end
