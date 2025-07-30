module AutoMLFlows
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
export mlfdriver, AutoMLFlow

const MLF = PYC.pynew()
const MLFC = PYC.pynew()

function __init__()
  PYC.pycopy!(MLF, PYC.pyimport("mlflow"))
end

mutable struct AutoMLFlow <: Workflow
  name::String
  model::Dict{Symbol,Any}

  function AutoMLFlow(args=Dict())
    default_args = Dict(
      :name => "automlflow",
      :projectname => "forecasting",
      :url => "http://localhost:8080",
      :description => "forecasting experiment",
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


function mlfdriver()
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
  # amlflow = AutoMLFlow()
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
  #amlflow = AutoMLFlow()
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

  MLF.end_run()
  amlflow = AutoMLFlow()
  MLF.start_run()
  df = getiris()
  X = df[:, 1:end-1]
  Yc = df[:, end] |> collect
  autoclass = AutoClassification()
  fit_transform!(autoclass, X, Yc)
  bestmodel = autoclass.model[:bestpipeline].model[:description]
  MLF.log_param("bestmodel",bestmodel)
  MLF.log_metric("bestperformance",autoclass.model[:performance].mean[1])
  serialize("./autoclass.bin",autoclass)
  MLF.log_artifact("./autoclass.bin")
  bestmodel_uri = MLF.get_artifact_uri(artifact_path = "autoclass.bin")
  pylocalpath=MLF.artifacts.download_artifacts(bestmodel_uri)
  bestmodel=deserialize(string(pylocalpath))
  Yh = transform!(bestmodel, X)
  println("accuracy = ",mean(Yh .== Yc))
  MLF.end_run()
  return autoclass
end

end
