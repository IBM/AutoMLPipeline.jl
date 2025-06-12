module CaretAnomalyDetectors

import PythonCall
const PYC = PythonCall

# standard included modules
using DataFrames: DataFrame
using Random
using ..AbsTypes
using ..Utils

import ..AbsTypes: fit, fit!, transform, transform!
export fit, fit!, transform, transform!
export CaretAnomalyDetector, caretanomalydetectors
export caretdriver

function caretanomalydetectors()
end

const CADX = PYC.pynew()
const CDATA = PYC.pynew()


function __init__()
  PYC.pycopy!(CADX, PYC.pyimport("pycaret.anomaly"))
  PYC.pycopy!(CDATA, PYC.pyimport("pycaret.datasets"))
end

const caretadlearner_dict = Dict{String,PYC.Py}(
  "abod" => CADX, "cluster" => CADX,
  "cof" => CADX, "iforest" => CADX,
  "histogram" => CADX, "knn" => CADX,
  "lof" => CADX, "svm" => CADX, "pca" => CADX,
  "mcd" => CADX, "sod" => CADX, "sos" => CADX
)

const caretexperiment_dict = Dict{String,PYC.Py}()
caretexperiment_dict["AnomalyExperiment"] = CADX

mutable struct CaretAnomalyDetector <: Learner
  name::String
  model::Dict{Symbol,Any}
  function CaretAnomalyDetector(args=Dict())
    default_args = Dict(
      :name => "caretad",
      :learner => "iforest",
      :experiment => "AnomalyExperiment",
      :output => "anomalies",
      :impl_args => Dict{Symbol,Any}()
    )
    cargs = nested_dict_merge(default_args, args)
    cargs[:name] = cargs[:name] * "_" * randstring(3)
    skl = cargs[:learner]
    if !(skl in keys(caretadlearner_dict))
      println("$skl is not supported.")
      println()
      caretanomalydetectors()
      error("Argument keyword error")
    end
    new(cargs[:name], cargs)
  end
end

function CaretAnomalyDetector(learner::String, args::Dict)
  CaretAnomalyDetector(Dict(:learner => learner, :name => learner, args...))
end

function CaretAnomalyDetector(learner::String; args...)
  CaretAnomalyDetector(Dict(:learner => learner, :name => learner, :impl_args => Dict(pairs(args))))
end

function fit!(adl::CaretAnomalyDetector, xx::DataFrame, ::Vector=[])::Nothing
  x = xx |> Array
  impl_args = copy(adl.model[:impl_args])
  expt = adl.model[:experiment]
  learner = adl.model[:learner]
  py_experiment = getproperty(caretexperiment_dict[expt], expt)()
  py_experiment.setup(x, session_id=123)
  py_experiment.create_model(learner)

  # save model
  adl.model[:adlearner] = py_experiment
  return nothing
end

function transform!(adl::CaretAnomalyDetector, xx::DataFrame)
  x = deepcopy(xx) |> Array
  learner = adl.model[:learner]
  py_experiment = adl.model[:adlearner]
  py_experiment.setup(x, session_id=123)
  clearner = py_experiment.create_model(learner)
  res = py_experiment.assign_model(clearner)
  return res.Anomaly |> PYC.PyArray |> Vector
end


function caretdriver()
  # load sample dataset
  #get_data = CDATA.get_data
  #data = get_data("anomaly")
  #caretad = CaretAnomalyDetector()
  #expt = caretad.model[:experiment]
  #learner = caretad.model[:learner]
  #println(expt)
  #println(learner)
  #aexpr = CADX.AnomalyExperiment()
  #aexpr.setup(data, session_id=123)
  #aexpr.create_model(learner)
  #aexpr.models()
  df = rand(100, 3) |> x -> DataFrame(x, :auto)
  caretad = CaretAnomalyDetector()
  fit!(caretad, df)
  transform!(caretad, df)

end


end
