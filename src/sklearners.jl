module SKLearners

using PyCall

# standard included modules
using DataFrames
using Random
using AutoMLPipeline.AbsTypes
using AutoMLPipeline.Utils

import AutoMLPipeline.AbsTypes: fit!, transform!
export fit!, transform!
export SKLearner


function __init__()
  global ENS=pyimport_conda("sklearn.ensemble") 
  global LM=pyimport_conda("sklearn.linear_model")
  global DA=pyimport_conda("sklearn.discriminant_analysis")
  global NN=pyimport_conda("sklearn.neighbors")
  global SVM=pyimport_conda("sklearn.svm")
  global TREE=pyimport_conda("sklearn.tree")
  global ANN=pyimport_conda("sklearn.neural_network")
  global GP=pyimport_conda("sklearn.gaussian_process")
  global KR=pyimport_conda("sklearn.kernel_ridge")
  global NB=pyimport_conda("sklearn.naive_bayes")
  global ISO=pyimport_conda("sklearn.isotonic")

  # Available scikit-learn learners.
  global learner_dict = Dict(
       "AdaBoostClassifier" => ENS.AdaBoostClassifier,
       "BaggingClassifier" => ENS.BaggingClassifier,
       "ExtraTreesClassifier" => ENS.ExtraTreesClassifier,
       "VotingClassifier" => ENS.VotingClassifier,
       "GradientBoostingClassifier" => ENS.GradientBoostingClassifier,
       "RandomForestClassifier" => ENS.RandomForestClassifier,
       "LDA" => DA.LinearDiscriminantAnalysis,
       "QDA" => DA.QuadraticDiscriminantAnalysis,
       "LogisticRegression" => LM.LogisticRegression,
       "PassiveAggressiveClassifier" => LM.PassiveAggressiveClassifier,
       "RidgeClassifier" => LM.RidgeClassifier,
       "RidgeClassifierCV" => LM.RidgeClassifierCV,
       "SGDClassifier" => LM.SGDClassifier,
       "KNeighborsClassifier" => NN.KNeighborsClassifier,
       "RadiusNeighborsClassifier" => NN.RadiusNeighborsClassifier,
       "NearestCentroid" => NN.NearestCentroid,
       "SVC" => SVM.SVC,
       "LinearSVC" => SVM.LinearSVC,
       "NuSVC" => SVM.NuSVC,
       "MLPClassifier" => ANN.MLPClassifier,
       "GaussianProcessClassifier" => GP.GaussianProcessClassifier,
       "DecisionTreeClassifier" => TREE.DecisionTreeClassifier,
       "GaussianNB" => NB.GaussianNB,
       "MultinomialNB" => NB.MultinomialNB,
       "ComplementNB" => NB.ComplementNB,
       "BernoulliNB" => NB.BernoulliNB,
       "SVR" => SVM.SVR,
       "Ridge" => LM.Ridge,
       "RidgeCV" => LM.RidgeCV,
       "Lasso" => LM.Lasso,
       "ElasticNet" => LM.ElasticNet,
       "Lars" => LM.Lars,
       "LassoLars" => LM.LassoLars,
       "OrthogonalMatchingPursuit" => LM.OrthogonalMatchingPursuit,
       "BayesianRidge" => LM.BayesianRidge,
       "ARDRegression" => LM.ARDRegression,
       "SGDRegressor" => LM.SGDRegressor,
       "PassiveAggressiveRegressor" => LM.PassiveAggressiveRegressor,
       "KernelRidge" => KR.KernelRidge,
       "KNeighborsRegressor" => NN.KNeighborsRegressor,
       "RadiusNeighborsRegressor" => NN.RadiusNeighborsRegressor,
       "GaussianProcessRegressor" => GP.GaussianProcessRegressor,
       "DecisionTreeRegressor" => TREE.DecisionTreeRegressor,
       "RandomForestRegressor" => ENS.RandomForestRegressor,
       "ExtraTreesRegressor" => ENS.ExtraTreesRegressor,
       "AdaBoostRegressor" => ENS.AdaBoostRegressor,
       "GradientBoostingRegressor" => ENS.GradientBoostingRegressor,
       "IsotonicRegression" => ISO.IsotonicRegression,
       "MLPRegressor" => ANN.MLPRegressor
      )
end

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
      println("keywords: ", keys(learner_dict))
      error("Argument keyword error")
    end
    new(cargs[:name],Dict(),cargs)
  end
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

function SKLearner(learner::String)
  SKLearner(Dict(:learner => learner))
end

function transform!(skl::SKLearner, xx::DataFrame)
  x = xx |> Array
  #return collect(skl.model[:predict](x))
  sklearner = skl.model[:sklearner]
  return collect(sklearner.predict(x))
end

end

