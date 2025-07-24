module SKAnomalyDetectors

import PythonCall
const PYC = PythonCall

# standard included modules
using DataFrames: DataFrame
using Random
using ..AbsTypes
using ..Utils

import ..AbsTypes: fit, fit!, transform, transform!
export fit, fit!, transform, transform!
export SKAnomalyDetector, skanomalydetectors
export skaddriver

const adlearner_dict = Dict{String,PYC.Py}()
const AENS = PYC.pynew()
const ACOV = PYC.pynew()
const ASVM = PYC.pynew()
const ASNN = PYC.pynew()

function __init__()
  PYC.pycopy!(AENS, PYC.pyimport("sklearn.ensemble"))
  PYC.pycopy!(ACOV, PYC.pyimport("sklearn.covariance"))
  PYC.pycopy!(ASVM, PYC.pyimport("sklearn.svm"))
  PYC.pycopy!(ASNN, PYC.pyimport("sklearn.neighbors"))
end

adlearner_dict["IsolationForest"] = AENS
adlearner_dict["EllipticEnvelope"] = ACOV
adlearner_dict["OneClassSVM"] = ASVM
adlearner_dict["LocalOutlierFactor"] = ASNN

function skanomalydetectors()
  detectors = keys(adlearner_dict) |> collect |> x -> sort(x, lt=(x, y) -> lowercase(x) < lowercase(y))
  println("syntax: SKAnomalyDetector(name::String, args::Dict=Dict())")
  println("where 'name' can be one of:")
  println()
  [print(learner, " ") for learner in detectors]
  println()
  println()
  println("and 'args' are the corresponding learner's initial parameters.")
  println("Note: Consult Scikitlearn's online help for more details about the learner's arguments.")
end

mutable struct SKAnomalyDetector <: Learner
  name::String
  model::Dict{Symbol,Any}
  function SKAnomalyDetector(args=Dict())
    default_args = Dict(
      :name => "skad",
      :learner => "IsolationForest",
      :output => "anomalies",
      :impl_args => Dict{Symbol,Any}()
    )
    cargs = nested_dict_merge(default_args, args)
    cargs[:name] = cargs[:name] * "_" * randstring(3)
    skl = cargs[:learner]
    if !(skl in keys(adlearner_dict))
      println("$skl is not supported.")
      println()
      skanomalydetectors()
      error("Argument keyword error")
    end
    new(cargs[:name], cargs)
  end
end

function SKAnomalyDetector(learner::String, args::Dict)
  SKAnomalyDetector(Dict(:learner => learner, :name => learner, args...))
end

function SKAnomalyDetector(learner::String; args...)
  SKAnomalyDetector(Dict(:learner => learner, :name => learner, :impl_args => Dict(pairs(args))))
end

function (adl::SKAnomalyDetector)(; objargs...)
  adl.model[:impl_args] = Dict(pairs(objargs))
  adname = adl.model[:learner]
  skobj = getproperty(adlearner_dict[adname], adname)
  newskobj = skobj(; objargs...)
  adl.model[:sklearner] = newskobj
  return adl
end

function fit!(adl::SKAnomalyDetector, xx::DataFrame, ::Vector=[])::Nothing
  x = xx |> Array
  impl_args = copy(adl.model[:impl_args])
  learner = adl.model[:learner]
  py_learner = getproperty(adlearner_dict[learner], learner)

  # Train
  modelobj = py_learner(; impl_args...)
  modelobj.fit(x)
  adl.model[:adlearner] = modelobj
  adl.model[:impl_args] = impl_args
  return nothing
end

function transform!(adl::SKAnomalyDetector, xx::DataFrame)::Vector
  x = deepcopy(xx) |> Array
  adlearner = adl.model[:adlearner]
  if adl.model[:learner] == "LocalOutlierFactor"
    res = adlearner.fit_predict(x)
    println(typeof(res))
  else
    res = adlearner.predict(x)
  end
  return res |> PYC.PyArray |> Vector |> x -> replace(x, 1 => 0, -1 => 1)
end

function skaddriver()
  Random.seed!(3)
  X = rand(100, 1) |> x -> DataFrame(x, :auto)
  clf1 = SKAnomalyDetector("IsolationForest")
  clf2 = SKAnomalyDetector("EllipticEnvelope")
  clf3 = SKAnomalyDetector("OneClassSVM")
  clf4 = SKAnomalyDetector("LocalOutlierFactor")
  res1 = fit_transform!(clf1, X)
  res2 = fit_transform!(clf2, X)
  res3 = fit_transform!(clf3, X)
  res4 = fit_transform!(clf4, X)
  return DataFrame(iso=res1, eli=res2, osvm=res3, lcl=res4)
end

end
