module AutoMLPipeline

using AMLPBase
using AMLPBase.AbsTypes
export Machine, Learner, Transformer, Workflow, Computer
export fit!, transform!,fit_transform!

using AMLPBase.Utils
export holdout, kfold, score, infer_eltype, nested_dict_to_tuples, 
       nested_dict_set!, nested_dict_merge, create_transformer,
       mergedict, getiris,
       skipmean,skipmedian,skipstd,
       aggregatorclskipmissing,
       getiris, getprofb

using AMLPBase.Baselines
export Baseline, Identity

using AMLPBase.BaseFilters
export Imputer,OneHotEncoder,Wrapper

using AMLPBase.FeatureSelectors
export FeatureSelector, CatFeatureSelector, NumFeatureSelector, CatNumDiscriminator

using AMLPBase.Normalizers
export Normalizer

using AMLPBase.DecisionTreeLearners
export PrunedTree,RandomForest,Adaboost

using AMLPBase.EnsembleMethods
export VoteEnsemble, StackEnsemble, BestLearner

using AMLPBase.CrossValidators
export crossvalidate

using AMLPBase.NARemovers
export NARemover

using AMLPBase.Pipelines
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
