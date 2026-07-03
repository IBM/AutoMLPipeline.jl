module OutlierDetectors

using DataFrames: DataFrame
using Random
using OutlierDetection
using ..AbsTypes
using ..Utils

import ..AbsTypes: fit, fit!, transform, transform!
export fit, fit!, transform, transform!
export OutlierDetector

mutable struct OutlierDetector <: Learner
  name::String
  model::Dict{Symbol,Any}

  function OutlierDetector(args=Dict{Symbol,Any}())
    default_args = Dict{Symbol,Any}(
      :name => "outlierdetector",
      :detector => nothing,
      :output => :score,
      :threshold => 0.95,
      :fit_args => Dict{Symbol,Any}(:verbosity => 0),
      :transform_args => Dict{Symbol,Any}()
    )
    cargs = nested_dict_merge(default_args, args)
    cargs[:name] = cargs[:name] * "_" * randstring(3)
    cargs[:detector] !== nothing || throw(ArgumentError("missing OutlierDetection detector"))
    cargs[:output] in (:score, :label) || throw(ArgumentError("output must be :score or :label"))
    new(cargs[:name], cargs)
  end
end

OutlierDetector(detector::OutlierDetection.OD.Detector; args...) = OutlierDetector(Dict{Symbol,Any}(:detector => detector, pairs(args)...))
OutlierDetector(detector::OutlierDetection.OD.Detector, args::Dict) = OutlierDetector(Dict{Symbol,Any}(:detector => detector, args...))

function (od::OutlierDetector)(; objargs...)
  od.model = nested_dict_merge(od.model, Dict{Symbol,Any}(pairs(objargs)))
  od.model[:output] in (:score, :label) || throw(ArgumentError("output must be :score or :label"))
  return od
end

_oddata(x::DataFrame) = permutedims(Matrix(x))

function fit!(od::OutlierDetector, xx::DataFrame, yy::Vector=[])::Nothing
  fit_args = copy(od.model[:fit_args])
  detector = od.model[:detector]
  data = _oddata(xx)
  fitresult = isempty(yy) ? OutlierDetection.fit(detector, data; fit_args...) : OutlierDetection.fit(detector, data, yy; fit_args...)
  od.model[:fitresult] = fitresult[1]
  od.model[:scores_train] = collect(fitresult[2])
  od.model[:fit_args] = fit_args
  return nothing
end

function fit(od::OutlierDetector, xx::DataFrame, y::Vector=[])::OutlierDetector
  fit!(od, xx, y)
  return deepcopy(od)
end

function transform!(od::OutlierDetector, xx::DataFrame)::Vector
  haskey(od.model, :fitresult) || throw(ArgumentError("OutlierDetector must be fit before transform"))
  detector = od.model[:detector]
  scores_test = collect(OutlierDetection.transform(detector, od.model[:fitresult], _oddata(xx)))
  if od.model[:output] == :score
    return scores_test
  end
  _, labels_test = OutlierDetection.classify_quantile(od.model[:threshold])((od.model[:scores_train], scores_test))
  return labels_test
end

transform(od::OutlierDetector, xx::DataFrame)::Vector = transform!(od, xx)

end
