module AutoOfflineRL

using DataFrames
using CSV

export fit, fit!, transform, transform!,fit_transform, fit_transform!
import AMLPipelineBase.AbsTypes: fit, fit!, transform, transform!

using AMLPipelineBase
using AMLPipelineBase: AbsTypes, Utils, BaselineModels, Pipelines
using AMLPipelineBase: BaseFilters, FeatureSelectors, DecisionTreeLearners
using AMLPipelineBase: EnsembleMethods, CrossValidators
using AMLPipelineBase: NARemovers

export Machine, Learner, Transformer, Workflow, Computer
export holdout, kfold, score, infer_eltype, nested_dict_to_tuples,
       nested_dict_set!, nested_dict_merge, create_transformer,
       mergedict, getiris,getprofb,
       skipmean,skipmedian,skipstd,
       aggregatorclskipmissing
export Baseline, Identity
export Imputer,OneHotEncoder,Wrapper
export PrunedTree,RandomForest,Adaboost
export VoteEnsemble, StackEnsemble, BestLearner
export FeatureSelector, CatFeatureSelector, NumFeatureSelector, CatNumDiscriminator
export crossvalidate
export NARemover
export @pipeline
export @pipelinex
export @pipelinez
export +, |>, *, |, >>
export Pipeline, ComboPipeline



#export RLMachine, RLOffline, WeeklyEpisodes


#Base.@kwdef struct WeeklyEpisodes <: RLMachine
#  _params::Dict = Dict()
#  _model::Dict = Dict()
#end

include("offlinerls.jl")
using .OfflineRLs
export DiscreteRLOffline, fit!
export driver, listdiscreateagents



end # module 
