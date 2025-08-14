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

end # module AutoAI
