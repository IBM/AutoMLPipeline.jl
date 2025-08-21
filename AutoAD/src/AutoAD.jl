module AutoAD

greet() = print("Hello World!")

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

# -------------
include("autoclassification.jl")
using .AutoClassifications
export AutoClassification

include("autoregression.jl")
using .AutoRegressions
export AutoRegression

include("skanomalydetector.jl")
using .SKAnomalyDetectors
export SKAnomalyDetector, skanomalydetectors
export skaddriver

include("caretanomalydetector.jl")
using .CaretAnomalyDetectors
export CaretAnomalyDetectors, CaretAnomalyDetector
export caretdriver


include("carettspredictor.jl")
using .CaretTSPredictors
export CaretTSPredictor, carettsdriver

include("autoanomalydetection.jl")
using .AutoAnomalyDetections
export AutoAnomalyDetection
export autoaddriver

include("automlflowclassification.jl")
using .AutoMLFlowClassifications
export AutoMLFlowClassification
export mlfcldriver

include("automlflowregression.jl")
using .AutoMLFlowRegressions
export AutoMLFlowRegression
export mlfregdriver

include("automlflowanomalydetection.jl")
using .AutoMLFlowAnomalyDetections
export AutoMLFlowAnomalyDetection
export mlfaddriver

end # module AutoAD
