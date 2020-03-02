module AutoMLPipeline

greet() = print("Hello World!")
export fit!, transform!, fit_transform!

include("abstracttypes.jl")
using .AbsTypes
export Machine, Computer, Workflow, Learner, Transformer

include("utils.jl")
using .Utils

include("basefilters.jl")
using .BaseFilters

include("pipelines.jl")
using .Pipelines
export @pipeline
export @pipelinex
export Pipeline, ComboPipeline 

include("featureselector.jl")
using .FeatureSelectors

include("skpreprocessor.jl")
using .SKPreprocessors
export SKPreprocessor, skpreprocessors

include("sklearners.jl")
using .SKLearners
export SKLearner, sklearners

include("decisiontree.jl")
using .DecisionTreeLearners

include("ensemble.jl")
using .EnsembleMethods

include("crossvalidator.jl")
using .CrossValidators
export crossvalidate

include("skcrossvalidator.jl")
using .SKCrossValidators
export crossvalidate

end # module
