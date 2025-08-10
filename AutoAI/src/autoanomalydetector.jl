module AutoAnomalyDetectors
# classification search blocks


using Distributed
using AutoMLPipeline
using DataFrames: DataFrame, nrow, rename!
using AutoMLPipeline: score
using Random
using Statistics
using ..AbsTypes
using ..Utils
using ..CaretAnomalyDetectors
import ..CaretAnomalyDetectors.caretadlearner_dict
using ..SKAnomalyDetectors

import ..AbsTypes: fit, fit!, transform, transform!
export fit, fit!, transform, transform!
export AutoAnomalyDetector, autoaddriver

# define customized type
mutable struct AutoAnomalyDetector <: Workflow
  name::String
  model::Dict{Symbol,Any}

  function AutoAnomalyDetector(args=Dict())
    default_args = Dict(
      :name => "autoad",
      :votepercent => 0.3,
      :impl_args => Dict()
    )
    cargs = nested_dict_merge(default_args, args)
    cargs[:name] = cargs[:name] * "_" * randstring(3)
    new(cargs[:name], cargs)
  end
end

function fit!(autodt::AutoAnomalyDetector, X::DataFrame, Y::Vector)
  return nothing
end

function fit(clfb::AutoAnomalyDetector, X::DataFrame, Y::Vector)
  return nothing
end

function transform!(autodt::AutoAnomalyDetector, X::DataFrame)
  # detect anomalies using caret
  dfres1 = DataFrame()
  for learner in keys(caretadlearner_dict)
    model = CaretAnomalyDetector(learner)
    res = fit_transform!(model, X)
    mname = string(learner)
    dfres1 = hcat(dfres1, DataFrame(mname => res; makeunique=true))
  end

  #detect anomalies using scikitlearn
  iso = SKAnomalyDetector("IsolationForest")
  eli = SKAnomalyDetector("EllipticEnvelope")
  osvm = SKAnomalyDetector("OneClassSVM")
  lcl = SKAnomalyDetector("LocalOutlierFactor")
  isores = fit_transform!(iso, X)
  elires = fit_transform!(eli, X)
  osvmres = fit_transform!(osvm, X)
  lclres = fit_transform!(lcl, X)
  dfres2 = DataFrame(iso=isores, eli=elires, osvm=osvmres, lcl=lclres)

  # combine results and get mean anomaly for each row
  mdf = hcat(dfres1, dfres2)
  mdfm = hcat(mdf, DataFrame(admean=mean.(eachrow(mdf))))
  # filter anomalies based on mean cut-off
  # cutoff = autodt.model[:votepercent]
  dfad = DataFrame()
  for cutoff in 0.1:0.1:1.0
    ndx = map(x -> x >= cutoff, mdfm.admean)
    dfad = hcat(dfad, DataFrame(n=ndx); makeunique=true)
  end
  names = map(x -> string(x), 0.1:0.1:1.0)
  rename!(dfad, names)
  return dfad
end

function transform(autodt::AutoAnomalyDetector, X::DataFrame)
end

function autoaddriver()
  autoaddt = AutoAnomalyDetector(Dict(:votepercent => 0.1))
  X = vcat(5 * cos.(-10:10), sin.(-30:30), 3 * cos.(-10:10), 2 * tan.(-10:10), sin.(-30:30)) |> x -> DataFrame([x], :auto)
  fit_transform!(autoaddt, X)
end


end
