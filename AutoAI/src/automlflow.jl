module AutoMLFlows
import PythonCall
const PYC = PythonCall

using DataFrames: DataFrame
using Random
using ..AbsTypes
using ..Utils

import ..AbsTypes: fit, fit!, transform, transform!
export fit, fit!, transform, transform!
export mlfdriver, AutoMLFlow

const MLF = PYC.pynew()
const MLFC = PYC.pynew()

function __init__()
  PYC.pycopy!(MLF, PYC.pyimport("mlflow"))
end

mutable struct AutoMLFlow
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
    mlflowclient = getproperty(MLF, "MlflowClient")
    mlflow_set_experiment = getproperty(MLF, "set_experiment")
    client = mlflowclient(cargs[:url])
    name = cargs[:name]
    experiment = client.search_experiments(filter_string="name = \'$name\'")
    if PYC.pylen(experiment) != 0
      mlflow_set_experiment(experiment[0].name)
    else
      theexperiment = client.create_experiment(name=name, tags=experiment_tags)
      cargs[:pyexperiment] = theexperiment
    end
    cargs[:pyclient] = client
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
  enable_system_metrics_logging = getproperty(MLF, "enable_system_metrics_logging")
  start_run = getproperty(MLF, "start_run")
  end_run = getproperty(MLF, "end_run")
  autolog = getproperty(MLF, "autolog")
  enable_system_metrics_logging()
  start_run()
  autolog()
  amlflow = AutoMLFlow()

  client = amlflow.model[:pyclient]
  exptname = amlflow.model[:name]
  pyexperiment = client.search_experiments(filter_string="name = \'$exptname\'")
  end_run()

  #for a in all_experiments
  #  println(a.name)
  #end

end

end
