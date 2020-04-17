module SKPreprocessors

using PyCall

# standard included modules
using DataFrames
using Random
using AutoMLPipeline.AbsTypes
using AutoMLPipeline.Utils

import AutoMLPipeline.AbsTypes: fit!, transform!
export fit!, transform!
export SKPreprocessor, skpreprocessors


function __init__()
  global DEC=pyimport_conda("sklearn.decomposition","scikit-learn") 
  global FS=pyimport_conda("sklearn.feature_selection","scikit-learn")
  global IMP=pyimport_conda("sklearn.impute","scikit-learn")
  global PREP=pyimport_conda("sklearn.preprocessing","scikit-learn")

  # Available scikit-learn learners.
  global preprocessor_dict = Dict(
     "DictionaryLearning" => DEC.DictionaryLearning,
     "FactorAnalysis" => DEC.FactorAnalysis,
     "FastICA" => DEC.FastICA,
     "IncrementalPCA" => DEC.IncrementalPCA,
     "KernelPCA" => DEC.KernelPCA,
     "LatentDirichletAllocation" => DEC.LatentDirichletAllocation,
     "MiniBatchDictionaryLearning" => DEC.MiniBatchDictionaryLearning,
     "MiniBatchSparsePCA" => DEC.MiniBatchSparsePCA,
     "NMF" => DEC.NMF,
     "PCA" => DEC.PCA, 
     "SparsePCA" => DEC.SparsePCA,
     "SparseCoder" => DEC.SparseCoder,
     "TruncatedSVD" => DEC.TruncatedSVD,
     "dict_learning" => DEC.dict_learning,
     "dict_learning_online" => DEC.dict_learning_online,
     "fastica" => DEC.fastica,
     "non_negative_factorization" => DEC.non_negative_factorization,
     "sparse_encode" => DEC.sparse_encode,
     "GenericUnivariateSelect" => FS.GenericUnivariateSelect,
     "SelectPercentile" => FS.SelectPercentile,
     "SelectKBest" => FS.SelectKBest,
     "SelectFpr" => FS.SelectFpr,
     "SelectFdr" => FS.SelectFdr,
     "SelectFromModel"  => FS.SelectFromModel,
     "SelectFwe" => FS.SelectFwe,
     "RFE" => FS.RFE,
     "RFECV" => FS.RFECV,
     "VarianceThreshold"  => FS.VarianceThreshold,
     "chi2" => FS.chi2,
     "f_classif"  => FS.f_classif,
     "f_regression" => FS.f_regression,
     "mutual_info_classif" => FS.mutual_info_classif,
     "mutual_info_regression" => FS.mutual_info_regression,
     "SimpleImputer" => IMP.SimpleImputer,
     #"IterativeImputer" => IMP.IterativeImputer,
     #"KNNImputer" => IMP.KNNImputer,
     "MissingIndicator" => IMP.MissingIndicator,
     "Binarizer" => PREP.Binarizer,
     "FunctionTransformer" => PREP.FunctionTransformer,
     "KBinsDiscretizer" => PREP.KBinsDiscretizer,
     "KernelCenterer" => PREP.KernelCenterer,
     "LabelBinarizer" => PREP.LabelBinarizer,
     "LabelEncoder" => PREP.LabelEncoder,
     "MultiLabelBinarizer" => PREP.MultiLabelBinarizer,
     "MaxAbsScaler" => PREP.MaxAbsScaler,
     "MinMaxScaler" => PREP.MinMaxScaler,
     "Normalizer" => PREP.Normalizer,
     "OneHotEncoder" => PREP.OneHotEncoder,
     "OrdinalEncoder" => PREP.OrdinalEncoder,
     "PolynomialFeatures" => PREP.PolynomialFeatures,
     "PowerTransformer" => PREP.PowerTransformer,
     "QuantileTransformer" => PREP.QuantileTransformer,
     "RobustScaler" => PREP.RobustScaler,
     "StandardScaler" => PREP.StandardScaler
     #"add_dummy_feature" => PREP.add_dummy_feature,
     #"binarize" => PREP.binarize,
     #"label_binarize" => PREP.label_binarize,
     #"maxabs_scale" => PREP.maxabs_scale,
     #"minmax_scale" => PREP.minmax_scale,
     #"normalize" => PREP.normalize,
     #"quantile_transform" => PREP.quantile_transform,
     #"robust_scale" => PREP.robust_scale,
     #"scale" => PREP.scale,
     #"power_transform" => PREP.power_transform
    )
end

"""
    SKPreprocessor(preprocessor::String,args::Dict=Dict())

A wrapper for Scikitlearn preprocessor functions. 
Invoking `skpreprocessors()` will list the acceptable 
and supported functions. Please check Scikitlearn
documentation for arguments to pass.

Implements `fit!` and `transform!`.
"""
mutable struct SKPreprocessor <: Transformer
  name::String
  model::Dict
  args::Dict

  function SKPreprocessor(args=Dict())
    default_args=Dict(
                      :name => "skprep",
                      :preprocessor => "PCA",
                      :impl_args => Dict(),
                      :autocomponent=>false
                     )
    cargs = nested_dict_merge(default_args, args)
    cargs[:name] = cargs[:name]*"_"*randstring(3)
    prep = cargs[:preprocessor]
    if !(prep in keys(preprocessor_dict)) 
      println("$prep is not supported.") 
      println()
      skpreprocessors()
      error("Argument keyword error")
    end
    new(cargs[:name],Dict(),cargs)
  end
end

function SKPreprocessor(prep::String,args::Dict=Dict())
  SKPreprocessor(Dict(:preprocessor => prep,:name=>prep,args...))
end

function skpreprocessors()
  processors = keys(preprocessor_dict) |> collect |> x-> sort(x,lt=(x,y)->lowercase(x)<lowercase(y))
  println("syntax: SKPreprocessor(name::String, args::Dict=Dict())")
  println("where *name* can be one of:")
  println()
  [print(processor," ") for processor in processors]
  println()
  println()
  println("and *args* are the corresponding preprocessor's initial parameters.")
  println("Note: Please consult Scikitlearn's online help for more details about the preprocessor's arguments.")
end

function fit!(skp::SKPreprocessor, x::DataFrame, y::Vector=[])
  features = x |> Array
  impl_args = copy(skp.args[:impl_args])
  autocomp = skp.args[:autocomponent]
  if autocomp == true
    cols = ncol(x)
    ncomponents = 1
    if cols > 0
      ncomponents = round(sqrt(cols),digits=0) |> Integer
      push!(impl_args,:n_components => ncomponents)
    end
  end
  preprocessor = skp.args[:preprocessor]
  py_preprocessor = preprocessor_dict[preprocessor]

  # Train model
  preproc = py_preprocessor(;impl_args...)
  preproc.fit(features)
  skp.model = Dict(
                   :skpreprocessor => preproc,
                   :impl_args => impl_args
                  )
end

function transform!(skp::SKPreprocessor, x::DataFrame)
  features = deepcopy(x) |> Array
  model=skp.model[:skpreprocessor]
  return collect(model.transform(features)) |> DataFrame
end

end

