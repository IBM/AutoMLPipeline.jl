module AutoMLPipeline

using AMLPipelineBase
using AMLPipelineBase.AbsTypes
export fit!, transform!,fit_transform!

using AMLPipelineBase
using AMLPipelineBase: AbsTypes, Utils, BaselineModels, Pipelines
using AMLPipelineBase: BaseFilters, FeatureSelectors, DecisionTreeLearners
using AMLPipelineBase: EnsembleMethods, CrossValidators
using AMLPipelineBase: NARemovers

export Machine, Learner, Transformer, Workflow, Computer
export holdout, kfold, score, infer_eltype, nested_dict_to_tuples, 
       nested_dict_set!, nested_dict_merge, create_transformer,
       mergedict, getiris, getprofb,
       skipmean,skipmedian,skipstd,
       aggregatorclskipmissing
export Baseline, Identity
export Imputer,OneHotEncoder,Wrapper
export PrunedTree,RandomForest,Adaboost
export VoteEnsemble, StackEnsemble, BestLearner
export FeatureSelector, CatFeatureSelector, NumFeatureSelector, CatNumDiscriminator
export crossvalidate
export NARemover
export @pipeline, @pipelinex
export +, |>, *, |
export Pipeline, ComboPipeline

import AMLPipelineBase.AbsTypes: fit!, transform!

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

export skoperator

function skoperator(name::String; args...)
   sklr = keys(SKLearners.learner_dict)
   skpr = keys(SKPreprocessors.preprocessor_dict)
   if name ∈ sklr
      obj = SKLearner(name; args...)
   elseif name ∈ skpr
      obj = SKPreprocessor(name; args...)
   else
      skoperator()
      throw(ArgumentError("$name does not exist"))
   end
end

function skoperator()
   sklr = keys(SKLearners.learner_dict)
   skpr = keys(SKPreprocessors.preprocessor_dict)
   println("Please choose among these pipeline elements:")
   println([sklr..., skpr...])
end

end # module
