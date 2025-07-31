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
using ..AutoRegressions
using ..SKAnomalyDetectors
using ..AutoMLPipeline: getiris

import ..AbsTypes: fit, fit!, transform, transform!
export fit, fit!, transform, transform!
export mlfdriver, AutoMLFlowClassification

const MLF = PYC.pynew()
const MLFC = PYC.pynew()

function __init__()
  PYC.pycopy!(MLF, PYC.pyimport("mlflow"))
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
      :impl_args => Dict()
    )
    cargs = nested_dict_merge(default_args, args)
    #cargs[:name] = cargs[:name] * "_" * randstring(3)
    experiment_tags = Dict(
      "projectname" => cargs[:projectname],
      "projecttype" => cargs[:projecttype],
      "notes" => cargs[:description]
    )
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
  # generate run name
  MLF.end_run()
  run_name = mlfcl.model[:name] * "_" * "fit" * "_" * randstring(3)
  mlfcl.model[:run_name] = run_name
  MLF.start_run(run_name=run_name)
  # automate classification
  autoclass = AutoClassification()
  fit_transform!(autoclass, X, Y)
  bestmodel = autoclass.model[:bestpipeline].model[:description]
  MLF.log_param("bestmodel", bestmodel)
  MLF.log_param("pipelines", autoclass.model[:dfpipelines].Description)
  MLF.log_metric("bestperformance", autoclass.model[:performance].mean[1])
  # save model in mlflow
  serialize("./autoclass.bin", autoclass)
  MLF.log_artifact("./autoclass.bin")
  # save model in memory
  mlfcl.model[:autoclass] = autoclass
  bestmodel_uri = MLF.get_artifact_uri(artifact_path="autoclass.bin")
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
  # download model artifact
  MLF.end_run()
  run_name = mlfcl.model[:name] * "_" * "transform" * "_" * randstring(3)
  mlfcl.model[:run_name] = run_name
  MLF.start_run(run_name=run_name)
  bestmodel_uri = mlfcl.model[:bestmodel_uri]
  pylocalpath = MLF.artifacts.download_artifacts(bestmodel_uri)
  bestmodel = deserialize(string(pylocalpath))
  Y = transform!(bestmodel, X)
  MLF.log_param("output", Y)
  MLF.end_run()
  return Y
end

function mlfdriver()
  df = getiris()
  X = df[:, 1:end-1]
  Y = df[:, end] |> collect

  mlfclass = AutoMLFlowClassification()
  Yc = fit_transform!(mlfclass, X, Y)
  println("accuracy = ", mean(Y .== Yc))

  #bestmodel_uri = "mlflow-artifacts:/159483400534934296/0f691026e1b74ba0a997f206d6334566/artifacts/autoclass.bin"
  ## caret new automl classification with saved artifact uri
  ##newmfclass = AutoMLFlowClassification(Dict(:bestmodel_uri => bestmodel_uri))
  ## create default automl classification type
  #newmfclass = AutoMLFlowClassification()
  ## add the existing best model uri location
  #newmfclass(; bestmodel_uri)
  #Yn = transform!(newmfclass, X)
  #println("accuracy = ", mean(Yn .== Y))

  #  #MLF.set_tracking_uri("http://localhost:8080")
  #  mlflowclient = getproperty(MLF, "MlflowClient")
  #  client = mlflowclient("http://localhost:8080")
  #  experiment_description = "forecasting experiment"
  #  experiment_tags = Dict(
  #    "projectname" => "forecasting",
  #    "projecttype" => "classification",
  #    "notes" => experiment_description
  #  )
  #  theexperiment = client.create_experiment(name="timeseries forecasting", tags=experiment_tags)
  #enable_system_metrics_logging = getproperty(MLF, "enable_system_metrics_logging")
  #autolog = getproperty(MLF, "autolog")
  #enable_system_metrics_logging()

  #  # extract py functions
  #  py_start_run = getproperty(MLF, "start_run")
  #  py_end_run = getproperty(MLF, "end_run")
  #  py_log_param = getproperty(MLF, "log_param")
  #
  #  #py_start_run()
  #  #autolog()
  #  id = "0"
  # amlflow = AutoMLFlowClassification()
  #  client = amlflow.model[:pyclient]
  #  run = client.create_run(id)
  #  #println(run)
  #  #client.log_param(run, key="datasize", value=150)
  #  df = getiris()
  #  #autoclass = classify(df)
  #  #println(autoclass.model[:bestpipeline])
  #
  #  client = amlflow.model[:pyclient]
  #  exptname = amlflow.model[:name]
  #  pyexperiment = client.search_experiments(filter_string="name = \'$exptname\'")
  #  py_end_run()
  #
  #  #for a in all_experiments
  #  #  println(a.name)
  #  #end
  #
  #py_set_experiment = getproperty(MLF, "set_experiment")
  #py_create_experiment = getproperty(MLF, "create_experiment")
  #py_start_run = getproperty(MLF, "start_run")
  #py_end_run = getproperty(MLF, "end_run")
  #py_log_param = getproperty(MLF, "log_param")
  #py_log_metric = getproperty(MLF, "log_metric")
  #py_set_tracking_uri = getproperty(MLF, "set_tracking_uri")

  #MLF.end_run()
  #amlflow = AutoMLFlowClassification()
  #run = MLF.start_run()
  #MLF.log_metric("mymetric", 2)
  #df = getiris()
  #autoclass = classify(df)
  ##println(autoclass.model[:description])
  #bestmodel = autoclass.model[:bestpipeline].model[:description]
  #MLF.log_param("bestmodel",bestmodel)
  #serialize("./autoclass.bin",autoclass)
  #MLF.log_artifact("./autoclass.bin")
  #MLF.end_run()
  #return 0
  #

  #MLF.end_run()
  #amlflow = AutoMLFlowClassification()
  #MLF.start_run()
  #df = getiris()
  #X = df[:, 1:end-1]
  #Yc = df[:, end] |> collect
  #autoclass = AutoClassification()
  #fit_transform!(autoclass, X, Yc)
  #bestmodel = autoclass.model[:bestpipeline].model[:description]
  #MLF.log_param("bestmodel", bestmodel)
  #MLF.log_param("pipelines", autoclass.model[:dfpipelines])
  #MLF.log_metric("bestperformance", autoclass.model[:performance].mean[1])
  #serialize("./autoclass.bin", autoclass)
  #MLF.log_artifact("./autoclass.bin")
  #bestmodel_uri = MLF.get_artifact_uri(artifact_path="autoclass.bin")
  #pylocalpath = MLF.artifacts.download_artifacts(bestmodel_uri)
  #bestmodel = deserialize(string(pylocalpath))
  #Yh = transform!(bestmodel, X)
  #println("accuracy = ", mean(Yh .== Yc))
  #MLF.end_run()
end

end
