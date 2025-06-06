module AutoAI

using Reexport
@reexport using AMLPipelineBase
using AMLPipelineBase.AbsTypes
export fit, fit!, transform, transform!, fit_transform, fit_transform!
import AMLPipelineBase.AbsTypes: fit!, transform!, fit, transform
using AMLPipelineBase: AbsTypes, Utils

@reexport using AutoMLPipeline

# -------------
include("clfsearchblock.jl")
using .ClfSearchBlocks
export ClfSearchBlock

greet() = print("Hello World!")
end # module AutoAI
