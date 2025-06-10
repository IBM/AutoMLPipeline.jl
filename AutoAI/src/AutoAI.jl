module AutoAI

using Reexport
@reexport using AMLPipelineBase
using AMLPipelineBase.AbsTypes
export fit, fit!, transform, transform!, fit_transform, fit_transform!
import AMLPipelineBase.AbsTypes: fit!, transform!, fit, transform
using AMLPipelineBase: AbsTypes, Utils
@reexport using AutoMLPipeline

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
export driver


greet() = print("Hello World!")
end # module AutoAI
