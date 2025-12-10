module AutoTS

greet() = print("Hello World!")

include("./main.jl")

using Reexport
@reexport using AutoMLPipeline
@reexport using AMLPipelineBase

using CSV
using DataFrames
using AMLPipelineBase.AbsTypes
export fit, fit!, transform, transform!, fit_transform, fit_transform!
import AMLPipelineBase.AbsTypes: fit!, transform!, fit, transform
using AMLPipelineBase: AbsTypes, Utils

export get_iris

function get_iris()
  iris = CSV.read(joinpath(Base.@__DIR__, "../../data", "iris.csv"), DataFrame)
  return iris
end

# -----------------------

include("carettspredictor.jl")
using .CaretTSPredictors
export CaretTSPredictor, carettsdriver

include("automlflowtsprediction.jl")
using .AutoMLFlowTSPredictions
export AutoMLFlowTSPrediction
export mlftsdriver

end # module AutoTS
