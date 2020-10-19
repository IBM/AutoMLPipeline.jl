module SKPreprocessors

using PyCall

# standard included modules
using DataFrames
using Random
using ..AbsTypes
using ..Utils

import ..AbsTypes: fit!, transform!
export fit!, transform!
export SKPreprocessor, skpreprocessors

const preprocessor_dict = Dict{String,PyObject}()

function __init__()
   DEC  = pyimport_conda("sklearn.decomposition","scikit-learn")
   FS   = pyimport_conda("sklearn.feature_selection","scikit-learn")
   IMP  = pyimport_conda("sklearn.impute","scikit-learn")
   PREP = pyimport_conda("sklearn.preprocessing","scikit-learn")

   # Available scikit-learn learners.
   preprocessor_dict["DictionaryLearning"]          = DEC.DictionaryLearning
   preprocessor_dict["FactorAnalysis"]              = DEC.FactorAnalysis
   preprocessor_dict["FastICA"]                     = DEC.FastICA
   preprocessor_dict["IncrementalPCA"]              = DEC.IncrementalPCA
   preprocessor_dict["KernelPCA"]                   = DEC.KernelPCA
   preprocessor_dict["LatentDirichletAllocation"]   = DEC.LatentDirichletAllocation
   preprocessor_dict["MiniBatchDictionaryLearning"] = DEC.MiniBatchDictionaryLearning
   preprocessor_dict["MiniBatchSparsePCA"]          = DEC.MiniBatchSparsePCA
   preprocessor_dict["NMF"]                         = DEC.NMF
   preprocessor_dict["PCA"]                         = DEC.PCA
   preprocessor_dict["SparsePCA"]                   = DEC.SparsePCA
   preprocessor_dict["SparseCoder"]                 = DEC.SparseCoder
   preprocessor_dict["TruncatedSVD"]                = DEC.TruncatedSVD
   preprocessor_dict["dict_learning"]               = DEC.dict_learning
   preprocessor_dict["dict_learning_online"]        = DEC.dict_learning_online
   preprocessor_dict["fastica"]                     = DEC.fastica
   preprocessor_dict["non_negative_factorization"]  = DEC.non_negative_factorization
   preprocessor_dict["sparse_encode"]               = DEC.sparse_encode
   preprocessor_dict["GenericUnivariateSelect"]     = FS.GenericUnivariateSelect
   preprocessor_dict["SelectPercentile"]            = FS.SelectPercentile
   preprocessor_dict["SelectKBest"]                 = FS.SelectKBest
   preprocessor_dict["SelectFpr"]                   = FS.SelectFpr
   preprocessor_dict["SelectFdr"]                   = FS.SelectFdr
   preprocessor_dict["SelectFromModel"]             = FS.SelectFromModel
   preprocessor_dict["SelectFwe"]                   = FS.SelectFwe
   preprocessor_dict["RFE"]                         = FS.RFE
   preprocessor_dict["RFECV"]                       = FS.RFECV
   preprocessor_dict["VarianceThreshold"]           = FS.VarianceThreshold
   preprocessor_dict["chi2"]                        = FS.chi2
   preprocessor_dict["f_classif"]                   = FS.f_classif
   preprocessor_dict["f_regression"]                = FS.f_regression
   preprocessor_dict["mutual_info_classif"]         = FS.mutual_info_classif
   preprocessor_dict["mutual_info_regression"]      = FS.mutual_info_regression
   preprocessor_dict["SimpleImputer"]               = IMP.SimpleImputer
   preprocessor_dict["MissingIndicator"]            = IMP.MissingIndicator
   preprocessor_dict["Binarizer"]                   = PREP.Binarizer
   preprocessor_dict["FunctionTransformer"]         = PREP.FunctionTransformer
   preprocessor_dict["KBinsDiscretizer"]            = PREP.KBinsDiscretizer
   preprocessor_dict["KernelCenterer"]              = PREP.KernelCenterer
   preprocessor_dict["LabelBinarizer"]              = PREP.LabelBinarizer
   preprocessor_dict["LabelEncoder"]                = PREP.LabelEncoder
   preprocessor_dict["MultiLabelBinarizer"]         = PREP.MultiLabelBinarizer
   preprocessor_dict["MaxAbsScaler"]                = PREP.MaxAbsScaler
   preprocessor_dict["MinMaxScaler"]                = PREP.MinMaxScaler
   preprocessor_dict["Normalizer"]                  = PREP.Normalizer
   preprocessor_dict["OneHotEncoder"]               = PREP.OneHotEncoder
   preprocessor_dict["OrdinalEncoder"]              = PREP.OrdinalEncoder
   preprocessor_dict["PolynomialFeatures"]          = PREP.PolynomialFeatures
   preprocessor_dict["PowerTransformer"]            = PREP.PowerTransformer
   preprocessor_dict["QuantileTransformer"]         = PREP.QuantileTransformer
   preprocessor_dict["RobustScaler"]                = PREP.RobustScaler
   preprocessor_dict["StandardScaler"]              = PREP.StandardScaler
   #"IterativeImputer" => IMP.IterativeImputer,
   #"KNNImputer" => IMP.KNNImputer,
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

