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

function fit!(mlfcl::AutoMLFlowRegression, X::DataFrame, Y::Vector)
  # end any running experiment
  # MLF.end_run()
  # generate run name
  run_name = mlfcl.model[:name] * "_" * "fit" * "_" * randstring(3)
  mlfcl.model[:run_name] = run_name
  MLF.set_experiment(mlfcl.model[:name])
  MLF.start_run(run_name=run_name)
  # automate classification
  autoclass = AutoRegression()
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

function fit(mlfcl::AutoMLFlowRegression, X::DataFrame, Y::Vector)
  mlfcopy = deepcopy(mlfcl)
  fit!(mlfcopy, X, Y)
  return mlfcopy
end

function transform!(mlfcl::AutoMLFlowRegression, X::DataFrame)
  # download model artifact
  #MLF.end_run()
  run_name = mlfcl.model[:name] * "_" * "transform" * "_" * randstring(3)
  mlfcl.model[:run_name] = run_name
  MLF.set_experiment(mlfcl.model[:name])
  MLF.start_run(run_name=run_name)
  bestmodel_uri = mlfcl.model[:bestmodel_uri]
  pylocalpath = MLF.artifacts.download_artifacts(bestmodel_uri)
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

  # test prediction using exisiting trained model from artifacts
  bestmodel_uri = "mlflow-artifacts:/302072302555887451/3c2e2a57955b4371ad894ed17bd8aeb0/artifacts/autoclass.bin"
  # caret new automl classification with saved artifact uri
  #newmfclass = AutoMLFlowRegression(Dict(:bestmodel_uri => bestmodel_uri))
  # create default automl classification type
  newmfclass = AutoMLFlowRegression()
  # add the existing best model uri location
  newmfclass(; bestmodel_uri)
  Yn = transform!(newmfclass, X)
  println("mse = ", mean((Y - Yn) .^ 2))

  #mlfclass = AutoMLFlowRegression()
  #Yc = fit_transform!(mlfclass, X, Y)
  #println("mse = ", mean((Y - Yc) .^ 2))
end

end
