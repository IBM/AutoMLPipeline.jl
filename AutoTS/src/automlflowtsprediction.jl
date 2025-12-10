module AutoMLFlowTSPredictions

using PDFmerger: append_pdf!
using Plots
using Statistics
using Serialization
import PythonCall
const PYC = PythonCall
using CSV

using DataFrames: DataFrame
using Random
using ..AbsTypes
using ..Utils
using ..CaretTSPredictors

import ..AbsTypes: fit, fit!, transform, transform!
export fit, fit!, transform, transform!
export mlftsdriver, AutoMLFlowTSPrediction

const MLF = PYC.pynew()
const REQ = PYC.pynew()

function __init__()
  PYC.pycopy!(MLF, PYC.pyimport("mlflow"))
  PYC.pycopy!(REQ, PYC.pyimport("requests"))
end

include("./mlflowutils.jl")

mutable struct AutoMLFlowTSPrediction <: Workflow
  name::String
  model::Dict{Symbol,Any}

  function AutoMLFlowTSPrediction(args=Dict())
    default_args = Dict(
      :name => "AutoTSPredictions",
      :projectname => "AutoTSPredictions",
      :url => "http://localhost:8080",
      :description => "Automated Timeseries Prediction",
      :projecttype => "tsprediction",
      :artifact_name => "AutoTSPredictionModel.bin",
      :impl_args => Dict(
        :name => "autots",
        :learner=>"auto",
        :forecast_horizon=>10
      )
    )
    cargs = nested_dict_merge(default_args, args)
    initmlflowcargs!(cargs)
    cargs[:automodel] = CaretTSPredictor(cargs[:impl_args])
    new(cargs[:name], cargs)
  end
end

function AutoMLFlowTSPrediction(name::String, args::Dict)
  AutoMLFlowTSPrediction(Dict(:name => name, args...))
end

function AutoMLFlowTSPrediction(name::String; args...)
  AutoMLFlowTSPrediction(Dict(Dict(pairs(args))...))
end

function (obj::AutoMLFlowTSPrediction)(; args...)
  model = obj.model
  cargs = nested_dict_merge(model, Dict(pairs(args)))
  obj.model = cargs
  return obj
end

function fit!(mlfas::AutoMLFlowTSPrediction, X::DataFrame, Y::Vector=[])::Nothing
  # start experiment run
  setupautofit!(mlfas)
  # automate prediction
  autots = mlfas.model[:automodel]
  tsoutput = fit_transform!(autots, X, Y)
  # save model in memory
  mlfas.model[:automodel] = autots
  # log info to mlflow
  MLF.log_param("TSOutput", tsoutput)
  MLF.log_metric("ForecastHorizon", autots.model[:forecast_horizon])
  # log artifacts and end experiment run
  logmlartifact(mlfas)
  return nothing
end

function fit(mlfas::AutoMLFlowTSPrediction, X::DataFrame, Y::Vector=[])::Nothing
  mlfcopy = deepcopy(mlfas)
  fit!(mlfcopy, X, Y)
  return mlfcopy
end

function plottroutput(mlfas::AutoMLFlowTSPrediction, Y::Union{Vector,DataFrame})
  data = Y
  votepercent = mlfas.model[:automodel].model[:votepercent]
  tmpdir = tempdir()
  println(tmpdir)
  artifact_plot = joinpath(tmpdir, "plots.pdf")
  artifact_allplots = joinpath(tmpdir, "allplots.pdf")
  rm(artifact_allplots, force=true)
  if votepercent == 0.0
    for ndx in 0.1:0.1:1.0
      strndx = string(ndx)
      coldata = data[:, strndx]
      ndx = findall(x -> x == true, coldata)
      Plots.plot(data[:,1], label="tsdata", title="TS Prediction")
      xlabel!("X")
      ylabel!("Y")
      plp = scatter!(ndx, data[:,1][ndx], label="prediction")
      savefig(plp, artifact_plot)
      append_pdf!(artifact_allplots, artifact_plot, cleanup=true)
    end
  else
    strndx = string(votepercent)
    coldata = data[:, strndx]
    ndx = findall(x -> x == true, coldata)
    Plots.plot(data[:,1], label="tsdata", title="TS Prediction")
    xlabel!("X")
    ylabel!("Y")
    scatter!(ndx, data[:,1][ndx], label="prediction")
    savefig(artifact_allplots)
  end
  MLF.log_artifact(artifact_allplots)
end

function transform!(mlfas::AutoMLFlowTSPrediction, X::DataFrame)
  # start experiment run
  Y = autotransform!(mlfas, X)
  # create plots and save them as mlfow artifacts
  # plottroutput(mlfas, Y)
  # end run
  MLF.end_run()
  return Y
end

function transform(mlfas::AutoMLFlowTSPrediction, X::DataFrame)
  mlfasc = deepcopy(mlfas)
  return transform!(mlfasc, X)
end

function mlftsdriver()
  url = "http://mlflow.home"
  url = "http://mlflow.isiath.duckdns.org:8082"
  url = "http://localhost:8081"

  X = CSV.read("./data/node_cpu_ratio_rate_5m_1d_1m.csv",DataFrame;header=false)

  #X = vcat(5 * cos.(-10:10), sin.(-30:30), 3 * cos.(-10:10), 2 * tan.(-10:10), sin.(-30:30)) |> x -> DataFrame([x], :auto)

  mlfas = AutoMLFlowTSPrediction(Dict(:url => url))
  pred=fit_transform!(mlfas, X)
  return pred
end

end
