module SKLearners

using PyCall

# standard included modules
using DataFrames
using Random
using ..AbsTypes
using ..Utils

import ..AbsTypes: fit, fit!, transform, transform!
export fit, fit!, transform, transform!
export SKLearner, sklearners

const learner_dict = Dict{String,PyObject}() 
const ENS   = PyNULL()
const LM    = PyNULL()
const DA    = PyNULL()
const NN    = PyNULL()
const SVM   = PyNULL()
const TREE  = PyNULL()
const ANN   = PyNULL()
const GP    = PyNULL()
const KR    = PyNULL()
const NB    = PyNULL()
const ISO   = PyNULL()

function __init__()
   copy!(ENS , pyimport_conda("sklearn.ensemble","scikit-learn"))
   copy!(LM  , pyimport_conda("sklearn.linear_model","scikit-learn"))
   copy!(DA  , pyimport_conda("sklearn.discriminant_analysis","scikit-learn"))
   copy!(NN  , pyimport_conda("sklearn.neighbors","scikit-learn"))
   copy!(SVM , pyimport_conda("sklearn.svm","scikit-learn"))
   copy!(TREE, pyimport_conda("sklearn.tree","scikit-learn"))
   copy!(ANN , pyimport_conda("sklearn.neural_network","scikit-learn"))
   copy!(GP  , pyimport_conda("sklearn.gaussian_process","scikit-learn"))
   copy!(KR  , pyimport_conda("sklearn.kernel_ridge","scikit-learn"))
   copy!(NB  , pyimport_conda("sklearn.naive_bayes","scikit-learn"))
   copy!(ISO , pyimport_conda("sklearn.isotonic","scikit-learn"))

   # Available scikit-learn learners.
   learner_dict["AdaBoostClassifier"]             = ENS
   learner_dict["BaggingClassifier"]              = ENS
   learner_dict["ExtraTreesClassifier"]           = ENS
   learner_dict["VotingClassifier"]               = ENS
   learner_dict["GradientBoostingClassifier"]     = ENS
   learner_dict["RandomForestClassifier"]         = ENS
   learner_dict["QuadraticDiscriminantAnalysis"] = DA
   learner_dict["LinearDiscriminantAnalysis"]     = DA
   learner_dict["LogisticRegression"]             = LM
   learner_dict["PassiveAggressiveClassifier"]    = LM
   learner_dict["RidgeClassifier"]                = LM
   learner_dict["RidgeClassifierCV"]              = LM
   learner_dict["SGDClassifier"]                  = LM
   learner_dict["KNeighborsClassifier"]           = NN
   learner_dict["RadiusNeighborsClassifier"]      = NN
   learner_dict["NearestCentroid"]                = NN
   learner_dict["SVC"]                            = SVM
   learner_dict["LinearSVC"]                      = SVM
   learner_dict["NuSVC"]                          = SVM
   learner_dict["MLPClassifier"]                  = ANN
   learner_dict["GaussianProcessClassifier"]      = GP
   learner_dict["DecisionTreeClassifier"]         = TREE
   learner_dict["GaussianNB"]                     = NB
   learner_dict["MultinomialNB"]                  = NB
   learner_dict["ComplementNB"]                   = NB
   learner_dict["BernoulliNB"]                    = NB
   learner_dict["SVR"]                            = SVM
   learner_dict["Ridge"]                          = LM
   learner_dict["RidgeCV"]                        = LM
   learner_dict["Lasso"]                          = LM
   learner_dict["ElasticNet"]                     = LM
   learner_dict["Lars"]                           = LM
   learner_dict["LassoLars"]                      = LM
   learner_dict["OrthogonalMatchingPursuit"]      = LM
   learner_dict["BayesianRidge"]                  = LM
   learner_dict["ARDRegression"]                  = LM
   learner_dict["SGDRegressor"]                   = LM
   learner_dict["PassiveAggressiveRegressor"]     = LM
   learner_dict["KernelRidge"]                    = KR
   learner_dict["KNeighborsRegressor"]            = NN
   learner_dict["RadiusNeighborsRegressor"]       = NN
   learner_dict["GaussianProcessRegressor"]       = GP
   learner_dict["DecisionTreeRegressor"]          = TREE
   learner_dict["RandomForestRegressor"]          = ENS
   learner_dict["ExtraTreesRegressor"]            = ENS
   learner_dict["AdaBoostRegressor"]              = ENS
   learner_dict["GradientBoostingRegressor"]      = ENS
   learner_dict["IsotonicRegression"]             = ISO
   learner_dict["MLPRegressor"]                   = ANN
end

"""
    SKLearner(learner::String, args::Dict=Dict())

A Scikitlearn wrapper to load the different machine learning models.
Invoking `sklearners()` will list the available learners. Please
consult Scikitlearn documentation for arguments to pass.

Implements `fit!` and `transform!`. 
"""
mutable struct SKLearner <: Learner
   name::String
   model::Dict{Symbol,Any}

   function SKLearner(args=Dict{Symbol,Any}())
      default_args=Dict{Symbol,Any}(
         :name => "sklearner",
         :output => :class,
         :learner => "LinearSVC",
         :impl_args => Dict{Symbol,Any}()
      )
      cargs = nested_dict_merge(default_args, args)
      cargs[:name] = cargs[:name]*"_"*randstring(3)
      skl = cargs[:learner]
      if !(skl in keys(learner_dict)) 
         println("$skl is not supported.") 
         println()
         sklearners()
         error("Argument keyword error")
      end
      new(cargs[:name],cargs)
   end
end

function SKLearner(learner::String, args::Dict)
   SKLearner(Dict(:learner => learner,:name=>learner, args...))
end

function SKLearner(learner::String; args...)
   SKLearner(Dict(:learner => learner,:name=>learner,:impl_args=>Dict(pairs(args))))
end

"""
    function sklearners()

List the available scikitlearn machine learners.
"""
function sklearners()
  learners = keys(learner_dict) |> collect |> x-> sort(x,lt=(x,y)->lowercase(x)<lowercase(y))
  println("syntax: SKLearner(name::String, args::Dict=Dict())")
  println("where 'name' can be one of:")
  println()
  [print(learner," ") for learner in learners]
  println()
  println()
  println("and 'args' are the corresponding learner's initial parameters.")
  println("Note: Consult Scikitlearn's online help for more details about the learner's arguments.")
end

function fit!(skl::SKLearner, xx::DataFrame, y::Vector)::Nothing
  x = xx |> Array
  impl_args  = copy(skl.model[:impl_args])
  learner    = skl.model[:learner]
  py_learner = getproperty(learner_dict[learner],learner)

  # Assign CombineML-specific defaults if required
  if learner == "RadiusNeighborsClassifier"
    if get(impl_args, :outlier_label, nothing) == nothing
      impl_options[:outlier_label] = labels[rand(1:size(labels, 1))]
    end
  end

  # Train
  modelobj = py_learner(;impl_args...)
  modelobj.fit(x,y)
  skl.model[:sklearner] = modelobj
  skl.model[:impl_args] = impl_args
  return nothing
end

function fit(skl::SKLearner, xx::DataFrame, y::Vector)::SKLearner
   fit!(skl,xx,y)
   return deepcopy(skl)
end

function transform!(skl::SKLearner, xx::DataFrame)::Vector
	x = deepcopy(xx) |> Array
  #return collect(skl.model[:predict](x))
  sklearner = skl.model[:sklearner]
  return collect(sklearner.predict(x))
end

transform(skl::SKLearner, xx::DataFrame)::Vector = transform!(skl,xx)

end

