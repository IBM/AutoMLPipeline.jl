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
export OneHotEncoder


include("featureselector.jl")
using .FeatureSelectors
export FeatureSelector, CatFeatureSelector, NumFeatureSelector, CatNumDiscriminator

include("skpreprocessor.jl")
using .SKPreprocessors
export SKPreprocessor, skpreprocessors

include("sklearners.jl")
using .SKLearners
export SKLearner, sklearners

include("decisiontree.jl")
using .DecisionTreeLearners
export PrunedTree, RandomForest, Adaboost

include("ensemble.jl")
using .EnsembleMethods
export VoteEnsemble, StackEnsemble, BestLearner

include("crossvalidator.jl")
using .CrossValidators
export crossvalidate

include("skcrossvalidator.jl")
using .SKCrossValidators
export crossvalidate

include("valdatefilters.jl")
using .ValDateFilters
export Matrifier,Dateifier
export DateValizer,DateValgator,DateValNNer,DateValMultiNNer
export CSVDateValReader, CSVDateValWriter, DateValLinearImputer
export BzCSVDateValReader


include("pipelines.jl")
using .Pipelines
export @pipeline
export @pipelinex
export Pipeline, ComboPipeline 

end # module
