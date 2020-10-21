module SKLearners

using PyCall

# standard included modules
using DataFrames
using Random
using ..AbsTypes
using ..Utils

import ..AbsTypes: fit!, transform!
export fit!, transform!
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
   learner_dict["AdaBoostClassifier"]          = ENS.AdaBoostClassifier
   learner_dict["BaggingClassifier"]           = ENS.BaggingClassifier
   learner_dict["ExtraTreesClassifier"]        = ENS.ExtraTreesClassifier
   learner_dict["VotingClassifier"]            = ENS.VotingClassifier
   learner_dict["GradientBoostingClassifier"]  = ENS.GradientBoostingClassifier
   learner_dict["RandomForestClassifier"]      = ENS.RandomForestClassifier
   learner_dict["LDA"]                         = DA.LinearDiscriminantAnalysis
   learner_dict["QDA"]                         = DA.QuadraticDiscriminantAnalysis
   learner_dict["LogisticRegression"]          = LM.LogisticRegression
   learner_dict["PassiveAggressiveClassifier"] = LM.PassiveAggressiveClassifier
   learner_dict["RidgeClassifier"]             = LM.RidgeClassifier
   learner_dict["RidgeClassifierCV"]           = LM.RidgeClassifierCV
   learner_dict["SGDClassifier"]               = LM.SGDClassifier
   learner_dict["KNeighborsClassifier"]        = NN.KNeighborsClassifier
   learner_dict["RadiusNeighborsClassifier"]   = NN.RadiusNeighborsClassifier
   learner_dict["NearestCentroid"]             = NN.NearestCentroid
   learner_dict["SVC"]                         = SVM.SVC
   learner_dict["LinearSVC"]                   = SVM.LinearSVC
   learner_dict["NuSVC"]                       = SVM.NuSVC
   learner_dict["MLPClassifier"]               = ANN.MLPClassifier
   learner_dict["GaussianProcessClassifier"]   = GP.GaussianProcessClassifier
   learner_dict["DecisionTreeClassifier"]      = TREE.DecisionTreeClassifier
   learner_dict["GaussianNB"]                  = NB.GaussianNB
   learner_dict["MultinomialNB"]               = NB.MultinomialNB
   learner_dict["ComplementNB"]                = NB.ComplementNB
   learner_dict["BernoulliNB"]                 = NB.BernoulliNB
   learner_dict["SVR"]                         = SVM.SVR
   learner_dict["Ridge"]                       = LM.Ridge
   learner_dict["RidgeCV"]                     = LM.RidgeCV
   learner_dict["Lasso"]                       = LM.Lasso
   learner_dict["ElasticNet"]                  = LM.ElasticNet
   learner_dict["Lars"]                        = LM.Lars
   learner_dict["LassoLars"]                   = LM.LassoLars
   learner_dict["OrthogonalMatchingPursuit"]   = LM.OrthogonalMatchingPursuit
   learner_dict["BayesianRidge"]               = LM.BayesianRidge
   learner_dict["ARDRegression"]               = LM.ARDRegression
   learner_dict["SGDRegressor"]                = LM.SGDRegressor
   learner_dict["PassiveAggressiveRegressor"]  = LM.PassiveAggressiveRegressor
   learner_dict["KernelRidge"]                 = KR.KernelRidge
   learner_dict["KNeighborsRegressor"]         = NN.KNeighborsRegressor
   learner_dict["RadiusNeighborsRegressor"]    = NN.RadiusNeighborsRegressor
   learner_dict["GaussianProcessRegressor"]    = GP.GaussianProcessRegressor
   learner_dict["DecisionTreeRegressor"]       = TREE.DecisionTreeRegressor
   learner_dict["RandomForestRegressor"]       = ENS.RandomForestRegressor
   learner_dict["ExtraTreesRegressor"]         = ENS.ExtraTreesRegressor
   learner_dict["AdaBoostRegressor"]           = ENS.AdaBoostRegressor
   learner_dict["GradientBoostingRegressor"]   = ENS.GradientBoostingRegressor
   learner_dict["IsotonicRegression"]          = ISO.IsotonicRegression
   learner_dict["MLPRegressor"]                = ANN.MLPRegressor
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
  model::Dict
  args::Dict

  function SKLearner(args=Dict())
    default_args=Dict(
       :name => "sklearner",
       :output => :class,
       :learner => "LinearSVC",
       :impl_args => Dict()
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
    new(cargs[:name],Dict(),cargs)
  end
end

function SKLearner(learner::String, args::Dict=Dict())
  SKLearner(Dict(:learner => learner,:name=>learner,  args...))
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

function fit!(skl::SKLearner, xx::DataFrame, y::Vector)
  x = xx |> Array
  impl_args = copy(skl.args[:impl_args])
  learner = skl.args[:learner]
  py_learner = learner_dict[learner]

  # Assign CombineML-specific defaults if required
  if learner == "RadiusNeighborsClassifier"
    if get(impl_args, :outlier_label, nothing) == nothing
      impl_options[:outlier_label] = labels[rand(1:size(labels, 1))]
    end
  end

  # Train
  modelobj = py_learner(;impl_args...)
  modelobj.fit(x,y)
  skl.model = Dict(
      :sklearner => modelobj,
      :impl_args => impl_args
     )
end


function transform!(skl::SKLearner, xx::DataFrame)
	x = deepcopy(xx) |> Array
  #return collect(skl.model[:predict](x))
  sklearner = skl.model[:sklearner]
  return collect(sklearner.predict(x))
end

end

