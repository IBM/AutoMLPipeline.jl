module SVMModels

using Random
using DataFrames
import LIBSVM

using ..AbsTypes
using ..BaseFilters
using ..Utils

import ..AbsTypes: fit!, transform!
export fit!, transform!
export SVMModel, svmlearners

const svm_dict = Dict{String,Any}()

function __init__()
   svm_dict["SVC"] = LIBSVM.SVC
   svm_dict["NuSVC"] = LIBSVM.NuSVC
   svm_dict["OneClassSVM"] = LIBSVM.OneClassSVM
   svm_dict["NuSVR"]=LIBSVM.NuSVR
   svm_dict["EpsilonSVR"] = LIBSVM.EpsilonSVR
   svm_dict["LinearSVC"] = LIBSVM.LinearSVC
end

"""
    SVMModel(
      Dict(
        :name => "svm",
        :svmtype => "SVC"
      )
    )

Wrapper for LIBSVM.jl for pipeline integration.

Implements `fit!` and `transform!`.
"""
mutable struct SVMModel <: Transformer
  name::String
  model::Dict{Symbol,Any}
  args::NamedTuple

  function SVMModel(args::Dict = Dict())
    default_args = Dict{Symbol,Any}(
	    :name => "libsvm",
       :learner => "SVC",
       :impl_args => Dict{Symbol,Any}(
               :tolerance => 0.001
       )
	 )
	 cargs=nested_dict_merge(default_args,args)
	 cargs[:name] = cargs[:name]*"_"*randstring(3)
    svmlearner = cargs[:learner]
    if !(svmlearner in keys(svm_dict))
       println("$svmlearner is not supported")
       println()
       svmlearners()
       throw(ArgumentError("not in dictionary"))
    end
    new(cargs[:name],Dict(:learner=>svmlearner),(;cargs[:impl_args]...))
  end
end


""" 
    SVMModel(svmt::String,opt=Dict())
 
Helper function
"""
function SVMModel(learner::String,opt::Dict)
   SVMModel(Dict(:learner=>learner,:impl_args=>opt))
end

""" 
    SVMModel(svmt::String;opt...)
 
Helper function
"""
function SVMModel(learner::String;opt...)
   SVMModel(Dict(:learner=>learner,:impl_args=>Dict(pairs(opt))))
end

function svmlearners()
   learners = keys(svm_dict) |> collect |> x-> sort(x,lt=(x,y)->lowercase(x)<lowercase(y))
   println("syntax: SVMModel(learner::String; args...)")
   println("where 'learner' can be one of:")
   println()
   [print(learner," ") for learner in learners]
   println()
   println()
   println("and 'args' are the corresponding learner's initial parameters.")
   println("Note: Consult LIBSVM.jl and LIBSVM github for more details about the learner's arguments.")
end


"""
   fit!(svm::SVMModel, features::DataFrame, labels::Vector=[])

Train the model using  SVM fit!

# Arguments
- `svm::SVMModel:` custom type
- `features::DataFrame`: input
- `labels::Vector=[]`: 
"""
function fit!(svm::SVMModel, features::DataFrame, labels::Vector=[])
  if features == DataFrame()
	 error("empty dataframe")
  end
  svmlearner = svm_dict[svm.model[:learner]]
  lv = convert(Vector,labels)
  featarray = convert(Array,features)
  impl_args = svm.args
  model = LIBSVM.fit!(svmlearner(;impl_args...),featarray,lv)
  svm.model[:svmmodel]=model
  nothing
end


"""
    transform!(svm::SVMModel, features::DataFrame)

Predict using svm model

# Arguments
- `svm::SVMModel`: custom type
- `features::DataFrame`: input
"""
function transform!(svm::SVMModel, features::DataFrame)
  if features == DataFrame()
	 error("empty dataframe")
  end
  featarray = convert(Array,features)
  model = svm.model[:svmmodel]
  pred = LIBSVM.predict(model,featarray)
  return pred
end

end
