module SKLearners

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
export SKLearner, sklearners

const learner_dict = Dict{String,PYC.Py}()
const ENS   = PYC.pynew()
const LM    = PYC.pynew()
const DA    = PYC.pynew()
const NN    = PYC.pynew()
const SVM   = PYC.pynew()
const TREE  = PYC.pynew()
const ANN   = PYC.pynew()
const GP    = PYC.pynew()
const KR    = PYC.pynew()
const NB    = PYC.pynew()
const ISO   = PYC.pynew()

function __init__()
   PYC.pycopy!(ENS , PYC.pyimport("sklearn.ensemble"))
   PYC.pycopy!(LM  , PYC.pyimport("sklearn.linear_model"))
   PYC.pycopy!(DA  , PYC.pyimport("sklearn.discriminant_analysis"))
   PYC.pycopy!(NN  , PYC.pyimport("sklearn.neighbors"))
   PYC.pycopy!(SVM , PYC.pyimport("sklearn.svm"))
   PYC.pycopy!(TREE, PYC.pyimport("sklearn.tree"))
   PYC.pycopy!(ANN , PYC.pyimport("sklearn.neural_network"))
   PYC.pycopy!(GP  , PYC.pyimport("sklearn.gaussian_process"))
   PYC.pycopy!(KR  , PYC.pyimport("sklearn.kernel_ridge"))
   PYC.pycopy!(NB  , PYC.pyimport("sklearn.naive_bayes"))
   PYC.pycopy!(ISO , PYC.pyimport("sklearn.isotonic"))

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

function (skl::SKLearner)(;objargs...)
   skl.model[:impl_args] = Dict(pairs(objargs))
   skname = skl.model[:learner] 
   skobj = getproperty(learner_dict[skname],skname)
   newskobj = skobj(;objargs...)
   skl.model[:sklearner] = newskobj
   return skl
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

function fit!(skl::SKLearner, xx::DataFrame, yy::Vector)::Nothing
    with_span("fit $(skl.model[:learner])") do 
        # normalize inputs
        x = xx |> Array
        y = yy
        skl.model[:predtype] = :numeric
        if !(eltype(yy) <: Real)
            y = yy |> Vector{String}
            skl.model[:predtype] = :alpha
        end

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
    end
    return nothing
end

function fit(skl::SKLearner, xx::DataFrame, y::Vector)::SKLearner
   fit!(skl,xx,y)
   return deepcopy(skl)
end

function transform!(skl::SKLearner, xx::DataFrame)::Vector
    with_span("transform $(skl.model[:learner])") do
        x = deepcopy(xx) |> Array
        sklearner = skl.model[:sklearner]
        res = sklearner.predict(x) 
        if skl.model[:predtype] == :numeric
            predn =  PYC.pyconvert(Vector{Float64},res) 
            return predn
        else
            predc =  PYC.pyconvert(Vector{String},res) 
            return predc
        end
    end
end

transform(skl::SKLearner, xx::DataFrame)::Vector = transform!(skl,xx)

end

