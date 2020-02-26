module AutoMLPipeline

greet() = print("Hello World!")
export fit!, transform!, fit_transform!

include("abstracttypes.jl")
using .AbsTypes

include("utils.jl")
using .Utils

include("basefilters.jl")
using .BaseFilters


include("pipelines.jl")
using .Pipelines
export @pipeline
export @pipelinex

include("featureselector.jl")
using .FeatureSelectors

include("skpreprocessor.jl")
using .SKPreprocessors

include("decisiontree.jl")
using .DecisionTreeLearners

include("ensemble.jl")
using .EnsembleMethods

include("crossvalidator.jl")
using .CrossValidators

include("skcrossvalidator.jl")
using .SKCrossValidators

end # module
