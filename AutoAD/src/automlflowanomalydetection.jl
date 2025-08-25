module AutoMLFlowAnomalyDetections
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
using ..AutoAnomalyDetections

import ..AbsTypes: fit, fit!, transform, transform!
export fit, fit!, transform, transform!
export mlfaddriver, AutoMLFlowAnomalyDetection

const MLF = PYC.pynew()
const REQ = PYC.pynew()

function __init__()
  PYC.pycopy!(MLF, PYC.pyimport("mlflow"))
  PYC.pycopy!(REQ, PYC.pyimport("requests"))
end

include("./mlflowutils.jl")

mutable struct AutoMLFlowAnomalyDetection <: Workflow
  name::String
  model::Dict{Symbol,Any}

  function AutoMLFlowAnomalyDetection(args=Dict())
    default_args = Dict(
      :name => "AutoAnomalyDetections",
      :projectname => "AutoAnomalyDetections",
      :url => "http://localhost:8080",
      :description => "Automated Anomaly Detection",
      :projecttype => "anomalydetection",
      :artifact_name => "AutoAnomalyDetectionModel.bin",
      :impl_args => Dict(
        :name => "autoad",
        :votepercent => 0.0, # output all votepercent if 0.0, otherwise get specific votepercent
      )
    )
    cargs = nested_dict_merge(default_args, args)
    initmlflowcargs!(cargs)
    cargs[:automodel] = AutoAnomalyDetection(cargs[:impl_args])
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
  # start experiment run
  setupautofit!(mlfad)
  # automate anomaly detection
  autoad = mlfad.model[:automodel]
  adoutput = fit_transform!(autoad, X, Y)
  # save model in memory
  mlfad.model[:automodel] = autoad
  # log info to mlflow
  MLF.log_param("ADOutput", adoutput)
  MLF.log_metric("votepercent", autoad.model[:votepercent])
  # log artifacts and end experiment run
  logmlartifact(mlfad)
end

function fit(mlfad::AutoMLFlowAnomalyDetection, X::DataFrame, Y::Vector)
  mlfcopy = deepcopy(mlfad)
  fit!(mlfcopy, X, Y)
  return mlfcopy
end

function plottroutput(mlfad::AutoMLFlowAnomalyDetection, Y::Union{Vector,DataFrame})
  data = Y
  votepercent = mlfad.model[:automodel].model[:votepercent]
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
      Plots.plot(data[:,1], label="tsdata", title="Anomaly voting cutoff=$strndx")
      xlabel!("X")
      ylabel!("Y")
      plp = scatter!(ndx, data[:,1][ndx], label="anomalous")
      savefig(plp, artifact_plot)
      append_pdf!(artifact_allplots, artifact_plot, cleanup=true)
    end
  else
    strndx = string(votepercent)
    coldata = data[:, strndx]
    ndx = findall(x -> x == true, coldata)
    Plots.plot(data[:,1], label="tsdata", title="Anomaly voting cutoff=$strndx")
    xlabel!("X")
    ylabel!("Y")
    scatter!(ndx, data[:,1][ndx], label="anomalous")
    savefig(artifact_allplots)
  end
  MLF.log_artifact(artifact_allplots)
end

function transform!(mlfad::AutoMLFlowAnomalyDetection, X::DataFrame)
  # start experiment run
  Y = autotransform!(mlfad, X)
  # create plots and save them as mlfow artifacts
  plottroutput(mlfad, Y)
  # end run
  MLF.end_run()
  votepercent = mlfad.model[:automodel].model[:votepercent]
  if votepercent == 0.0
    return Y
  else
    strndx = string(votepercent)
    return Y[:, [strndx]]
  end
end

function transform(mlfad::AutoMLFlowAnomalyDetection, X::DataFrame)
  mlfadc = deepcopy(mlfad)
  return transform!(mlfadc, X)
end

function mlfaddriver()
  url = "http://localhost:8081"
  url = "http://mlflow.home"
  url = "http://mlflow.isiath.duckdns.org:8082"

  X = CSV.read("./data/node_cpu_ratio_rate_5m_1d_1m.csv",DataFrame;header=false)

  #X = vcat(5 * cos.(-10:10), sin.(-30:30), 3 * cos.(-10:10), 2 * tan.(-10:10), sin.(-30:30)) |> x -> DataFrame([x], :auto)

  # test all voting percent
  mlfad = AutoMLFlowAnomalyDetection(Dict(:url => url))
  Yc = fit_transform!(mlfad, X)
  println(Yc |> x -> first(x, 5))

  #  # test specific votepercent
  #  mlvad = AutoMLFlowAnomalyDetection(Dict(:url => url, :impl_args => Dict(:votepercent => 0.3)))
  #  Yc = fit_transform!(mlvad, X)
  #  println(Yc |> x -> first(x, 5))
  #
  #  # override default votepercent
  #  mlfad = AutoMLFlowAnomalyDetection(Dict(:url => url))
  #  mlfad.model[:automodel](; votepercent=0.5)
  #  Yc = fit_transform!(mlfad, X)
  #  println(Yc |> x -> first(x, 5))
  #
  #
  #  mlfad = AutoMLFlowAnomalyDetection(Dict(:url => url))
  #  mlfad.model[:automodel](; votepercent=0.0)
  #  Yc = fit_transform!(mlfad, X)
  #  println(Yc |> x -> first(x, 5))
  #
  #  ## test prediction using exisiting trained model from artifacts
  #  #### alternative 1 to use trained model for transform
  #  mlvad = AutoMLFlowAnomalyDetection(Dict(:url => url))
  #  Yc = fit_transform!(mlvad, X)
  #  run_id = mlvad.model[:run_id]
  #  newmlad = AutoMLFlowAnomalyDetection(Dict(:run_id => run_id, :url => url, :impl_args => Dict(:votepercent => 0.5)))
  #  newmlad.model[:automodel](; votepercent=0.2)
  #  Yn = transform!(newmlad, X)
  #  println(Yn |> x -> first(x, 5))
  #
  #  ## alternative 2 to use trained model for transform
  #  mlvad = AutoMLFlowAnomalyDetection(Dict(:url => url))
  #  Yc = fit_transform!(mlvad, X)
  #  run_id = mlvad.model[:run_id]
  #  votepercent = 0.3
  #  newmlad = AutoMLFlowAnomalyDetection(Dict(:url => url))
  #  newmlad(; run_id)
  #  newmlad.model[:automodel](; votepercent)
  #  Yn = transform!(newmlad, X)
  #  println(Yn |> x -> first(x, 5))

  return nothing
end

end
