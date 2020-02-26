module SKPreprocessors

# standard included modules
using DataFrames
using Random
using AutoMLPipeline.AbsTypes
using AutoMLPipeline.Utils

import AutoMLPipeline.AbsTypes: fit!, transform!
export fit!, transform!
export SKPreprocessor

using PyCall

function __init__()
  global DEC=pyimport("sklearn.decomposition") 
  global FS=pyimport("sklearn.feature_selection")
  global IMP=pyimport("sklearn.impute")
  global PREP=pyimport("sklearn.preprocessing")

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

mutable struct SKPreprocessor <: Transformer
  name::String
  model::Dict
  args::Dict

  function SKPreprocessor(args=Dict())
    default_args=Dict(
                      :name => "skprep",
                      :preprocessor => "PCA",
                      :impl_args => Dict()
                     )
    cargs = nested_dict_merge(default_args, args)
    cargs[:name] = cargs[:name]*"_"*randstring(3)
    prep = cargs[:preprocessor]
    if !(prep in keys(preprocessor_dict)) 
      println("$prep is not supported.") 
      println("keywords: ", keys(preprocessor_dict))
      error("Argument keyword error")
    end
    new(cargs[:name],Dict(),cargs)
  end
end

function SKPreprocessor(prep::String)
  SKPreprocessor(Dict(:preprocessor => prep))
end

function fit!(skp::SKPreprocessor, x::DataFrame, y::Vector=[])
  features = x |> Array
  impl_args = copy(skp.args[:impl_args])
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
  features = x |> Array
  model=skp.model[:skpreprocessor]
  return collect(model.transform(features)) |> DataFrame
end

end

