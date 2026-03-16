module AutoMLPipeline

using AMLPipelineBase
using AMLPipelineBase.AbsTypes
export fit, fit!, transform, transform!, fit_transform, fit_transform!

using AMLPipelineBase
using AMLPipelineBase: AbsTypes, Utils, BaselineModels, Pipelines
using AMLPipelineBase: BaseFilters, FeatureSelectors, DecisionTreeLearners
using AMLPipelineBase: EnsembleMethods, CrossValidators
using AMLPipelineBase: NARemovers

export Machine, Learner, Transformer, Workflow, Computer
export holdout, kfold, score, infer_eltype, nested_dict_to_tuples,
  nested_dict_set!, nested_dict_merge,
  mergedict, getiris, getprofb,
  skipmean, skipmedian, skipstd,
  aggregatorclskipmissing,
  find_catnum_columns,
  train_test_split


export Baseline, Identity
export Imputer, OneHotEncoder, Wrapper
export PrunedTree, RandomForest, Adaboost
export VoteEnsemble, StackEnsemble, BestLearner
export FeatureSelector, CatFeatureSelector, NumFeatureSelector, CatNumDiscriminator
export crossvalidate
export NARemover
export @pipeline, @pipelinex
export +, |>, *, |, >>
export Pipeline, ComboPipeline

import AMLPipelineBase.AbsTypes: fit!, transform!

# --------------------------------------------

include("skpreprocessor.jl")
using .SKPreprocessors
export SKPreprocessor, skpreprocessors

include("sklearners.jl")
using .SKLearners
export SKLearner, sklearners

include("xgboostlearners.jl")
using .XGBoostLearners
export XGBoostLearner, xgboostlearners

include("evotreeslearners.jl")
using .EvoTreesLearners
export EvoTreesLearner, evotreeslearners

include("skcrossvalidator.jl")
using .SKCrossValidators
export crossvalidate

export skoperator, xgboostoperator, evotreesoperator

function skoperator(name::String; args...)::Machine
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
  return obj
end

function skoperator()
  sklr = keys(SKLearners.learner_dict)
  skpr = keys(SKPreprocessors.preprocessor_dict)
  println("Please choose among these pipeline elements:")
  println([sklr..., skpr...])
end

function xgboostoperator(name::String; args...)::Machine
  xgblr = keys(XGBoostLearners.learner_dict)
  if name ∈ xgblr
    obj = XGBoostLearner(name; args...)
  else
    xgboostoperator()
    throw(ArgumentError("$name does not exist"))
  end
  return obj
end

function xgboostoperator()
  xgblr = keys(XGBoostLearners.learner_dict)
  println("Please choose among these XGBoost learners:")
  println([xgblr...])
end


function evotreesoperator(name::String; args...)::Machine
  evolr = keys(EvoTreesLearners.learner_dict)
  if name ∈ evolr
    obj = EvoTreesLearner(name; args...)
  else
    evotreesoperator()
    throw(ArgumentError("$name does not exist"))
  end
  return obj
end

function evotreesoperator()
  evolr = keys(EvoTreesLearners.learner_dict)
  println("Please choose among these EvoTrees learners:")
  println([evolr...])
end

end # module
