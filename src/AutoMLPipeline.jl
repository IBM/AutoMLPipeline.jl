module AutoMLPipeline

using AMLPipelineBase
using AMLPipelineBase.AbsTypes
export Machine, Learner, Transformer, Workflow, Computer
export fit!, transform!,fit_transform!

using AMLPipelineBase.Utils
export holdout, kfold, score, infer_eltype, nested_dict_to_tuples, 
       nested_dict_set!, nested_dict_merge, create_transformer,
       mergedict, getiris,
       skipmean,skipmedian,skipstd,
       aggregatorclskipmissing,
       getiris, getprofb

using AMLPipelineBase.BaselineModels
export Baseline, Identity

using AMLPipelineBase.BaseFilters
export Imputer,OneHotEncoder,Wrapper

using AMLPipelineBase.FeatureSelectors
export FeatureSelector, CatFeatureSelector, NumFeatureSelector, CatNumDiscriminator

using AMLPipelineBase.DecisionTreeLearners
export PrunedTree,RandomForest,Adaboost

using AMLPipelineBase.EnsembleMethods
export VoteEnsemble, StackEnsemble, BestLearner

using AMLPipelineBase.CrossValidators
export crossvalidate

using AMLPipelineBase.NARemovers
export NARemover

using AMLPipelineBase.Pipelines
export @pipeline @pipelinex
export Pipeline, ComboPipeline

# --------------------------------------------

include("skpreprocessor.jl")
using .SKPreprocessors
export SKPreprocessor, skpreprocessors

include("sklearners.jl")
using .SKLearners
export SKLearner, sklearners

include("skcrossvalidator.jl")
using .SKCrossValidators
export crossvalidate

end # module
