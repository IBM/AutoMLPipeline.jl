module SKPreprocessors

import PythonCall
const PYC=PythonCall

# standard included modules
using DataFrames
using Random
using ..AbsTypes
using ..Utils

using OpenTelemetry
using Term
using Logging

import ..AbsTypes: fit, fit!, transform, transform!
export fit, fit!, transform, transform!
export SKPreprocessor, skpreprocessors

const preprocessor_dict = Dict{String,PYC.Py}()
const DEC  = PYC.pynew()
const FS   = PYC.pynew()
const IMP  = PYC.pynew()
const PREP = PYC.pynew()


function __init__()
   PYC.pycopy!(DEC , PYC.pyimport("sklearn.decomposition"))
   PYC.pycopy!(FS  , PYC.pyimport("sklearn.feature_selection",))
   PYC.pycopy!(IMP , PYC.pyimport("sklearn.impute"))
   PYC.pycopy!(PREP, PYC.pyimport("sklearn.preprocessing"))

   # Available scikit-learn learners.
   preprocessor_dict["DictionaryLearning"]          = DEC
   preprocessor_dict["FactorAnalysis"]              = DEC
   preprocessor_dict["FastICA"]                     = DEC
   preprocessor_dict["IncrementalPCA"]              = DEC
   preprocessor_dict["KernelPCA"]                   = DEC
   preprocessor_dict["LatentDirichletAllocation"]   = DEC
   preprocessor_dict["MiniBatchDictionaryLearning"] = DEC
   preprocessor_dict["MiniBatchSparsePCA"]          = DEC
   preprocessor_dict["NMF"]                         = DEC
   preprocessor_dict["PCA"]                         = DEC
   preprocessor_dict["SparsePCA"]                   = DEC
   preprocessor_dict["SparseCoder"]                 = DEC
   preprocessor_dict["TruncatedSVD"]                = DEC
   preprocessor_dict["dict_learning"]               = DEC
   preprocessor_dict["dict_learning_online"]        = DEC
   preprocessor_dict["fastica"]                     = DEC
   preprocessor_dict["non_negative_factorization"]  = DEC
   preprocessor_dict["sparse_encode"]               = DEC
   preprocessor_dict["GenericUnivariateSelect"]     = FS
   preprocessor_dict["SelectPercentile"]            = FS
   preprocessor_dict["SelectKBest"]                 = FS
   preprocessor_dict["SelectFpr"]                   = FS
   preprocessor_dict["SelectFdr"]                   = FS
   preprocessor_dict["SelectFromModel"]             = FS
   preprocessor_dict["SelectFwe"]                   = FS
   preprocessor_dict["RFE"]                         = FS
   preprocessor_dict["RFECV"]                       = FS
   preprocessor_dict["VarianceThreshold"]           = FS
   preprocessor_dict["chi2"]                        = FS
   preprocessor_dict["f_classif"]                   = FS
   preprocessor_dict["f_regression"]                = FS
   preprocessor_dict["mutual_info_classif"]         = FS
   preprocessor_dict["mutual_info_regression"]      = FS
   preprocessor_dict["SimpleImputer"]               = IMP
   preprocessor_dict["MissingIndicator"]            = IMP
   preprocessor_dict["Binarizer"]                   = PREP
   preprocessor_dict["FunctionTransformer"]         = PREP
   preprocessor_dict["KBinsDiscretizer"]            = PREP
   preprocessor_dict["KernelCenterer"]              = PREP
   preprocessor_dict["LabelBinarizer"]              = PREP
   preprocessor_dict["LabelEncoder"]                = PREP
   preprocessor_dict["MultiLabelBinarizer"]         = PREP
   preprocessor_dict["MaxAbsScaler"]                = PREP
   preprocessor_dict["MinMaxScaler"]                = PREP
   preprocessor_dict["Normalizer"]                  = PREP
   preprocessor_dict["OneHotEncoder"]               = PREP
   preprocessor_dict["OrdinalEncoder"]              = PREP
   preprocessor_dict["PolynomialFeatures"]          = PREP
   preprocessor_dict["PowerTransformer"]            = PREP
   preprocessor_dict["QuantileTransformer"]         = PREP
   preprocessor_dict["RobustScaler"]                = PREP
   preprocessor_dict["StandardScaler"]              = PREP
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
   model::Dict{Symbol,Any}

   function SKPreprocessor(args=Dict())
      default_args=Dict(
         :name => "skprep",
         :preprocessor => "PCA",
         :autocomponent=>false,
         :impl_args => Dict()
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
      new(cargs[:name],cargs)
   end
end

function SKPreprocessor(prep::String,args::Dict)
  SKPreprocessor(Dict(:preprocessor => prep,:name=>prep,args...))
end

function SKPreprocessor(prep::String; args...)
   SKPreprocessor(Dict(:preprocessor => prep,:name=>prep,:impl_args=>Dict(pairs(args))))
end

function (skp::SKPreprocessor)(;objargs...)
   skp.model[:impl_args] = Dict(pairs(objargs))
   prepname = skp.model[:preprocessor]
   skobj = getproperty(preprocessor_dict[prepname],prepname)
   newskobj = skobj(;objargs...)
   skp.model[:skpreprocessor] = newskobj
   return skp
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

function fit!(skp::SKPreprocessor, x::DataFrame, yc::Vector=[])::Nothing
    with_span("fit skp") do
        features = x |> Array
        y = yc
        #if !(eltype(yc) <: Real)
        #   y = yc |> Vector{String}
        #end

        impl_args = copy(skp.model[:impl_args])
        autocomp = skp.model[:autocomponent]
        if autocomp == true
            cols = ncol(x)
            ncomponents = 1
            if cols > 0
                ncomponents = round(sqrt(cols),digits=0) |> Integer
                push!(impl_args,:n_components => ncomponents)
            end
        end
        preprocessor = skp.model[:preprocessor]
        py_preprocessor = getproperty(preprocessor_dict[preprocessor],preprocessor)

        # Train model
        preproc = py_preprocessor(;impl_args...)
        preproc.fit(features)
        skp.model[:skpreprocessor] = preproc
        skp.model[:impl_args] = impl_args
    end
    return nothing
end

function fit(skp::SKPreprocessor, x::DataFrame, y::Vector=[])::SKPreprocessor
   fit!(skp,x,y)
   return deepcopy(skp)
end

function transform!(skp::SKPreprocessor, x::DataFrame)::DataFrame
    with_span("transform skp") do
        features = deepcopy(x) |> Array
        model=skp.model[:skpreprocessor]
        res = (model.transform(features))
        myres=PYC.pyconvert(Matrix,res) |> x->DataFrame(x,:auto)
        return myres
    end
end

transform(skp::SKPreprocessor, x::DataFrame)::DataFrame = transform!(skp,x)

end

