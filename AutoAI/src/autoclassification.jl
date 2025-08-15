module AutoClassifications
# classification search blocks


using Distributed
using AutoMLPipeline
using DataFrames: DataFrame
using AutoMLPipeline: score
using Random
using ..AbsTypes
using ..Utils

import ..AbsTypes: fit, fit!, transform, transform!
export fit, fit!, transform, transform!
export AutoClassification

include("./pipelinesearch.jl")

# define learners
const rfc = SKLearner("RandomForestClassifier", Dict(:name => "rfc"))
const adac = SKLearner("AdaBoostClassifier", Dict(:name => "adac"))
const gbc = SKLearner("GradientBoostingClassifier", Dict(:name => "gbc"))
const lsvc = SKLearner("LinearSVC", Dict(:name => "lsvc"))
const rbfsvc = SKLearner("SVC", Dict(:name => "rbfsvc"))
const dtc = SKLearner("DecisionTreeClassifier", Dict(:name => "dtc"))
const etc = SKLearner("ExtraTreesClassifier", Dict(:name => "etc"))
const ridgec = SKLearner("RidgeClassifier", Dict(:name => "ridgec"))
const sgdc = SKLearner("SGDClassifier", Dict(:name => "sgdc"))
#const gp     = SKLearner("GaussianProcessClassifier",Dict(:name =>"gp"))
const bgc = SKLearner("BaggingClassifier", Dict(:name => "bgc"))
const pac = SKLearner("PassiveAggressiveClassifier", Dict(:name => "pac"))

const _glearnerdict = Dict("rfc" => rfc, "gbc" => gbc,
  "lsvc" => lsvc, "rbfsvc" => rbfsvc, "adac" => adac,
  "dtc" => dtc, "etc" => etc, "ridgec" => ridgec,
  "sgdc" => sgdc, "bgc" => bgc, "pac" => pac
)

# define customized type
mutable struct AutoClassification <: Workflow
  name::String
  model::Dict{Symbol,Any}

  function AutoClassification(args=Dict())
    default_args = Dict(
      :name => "autoclass",
      :complexity => "low",
      :prediction_type => "classification",
      :nfolds => 3,
      :metric => "balanced_accuracy_score",
      :nworkers => 5,
      :learners => ["rfc", "rbfsvc", "gbc", "adac"],
      :scalers => ["norm", "pt", "mx", "std", "rb", "pt", "noop"],
      :extractors => ["pca", "ica", "fa", "noop"],
      :sortrev => true,
      :impl_args => Dict()
    )
    cargs = nested_dict_merge(default_args, args)
    cargs[:name] = cargs[:name] * "_" * randstring(3)
    learners = cargs[:learners]
    for learner in learners
      if !(learner in keys(_glearnerdict))
        println("$learner is not supported.")
        println()
        listclasslearners()
        error("Argument keyword error")
      end
    end
    new(cargs[:name], cargs)
  end
end

function AutoClassification(learners::Vector{String}, args::Dict)
  AutoClassification(Dict(:learners => learners, args...))
end

function AutoClassification(learners::Vector{String}; args...)
  AutoClassification(Dict(:learners => learners, :impl_args => Dict(pairs(args))))
end

function listclasslearners()
  println("Use available learners:")
  [print(learner, " ") for learner in keys(_glearnerdict)]
  println()
end

function fit!(autoclass::AutoClassification, X::DataFrame, Y::Vector)
  strscalers = autoclass.model[:scalers]
  strextractors = autoclass.model[:extractors]
  strlearners = autoclass.model[:learners]

  # get objects from dictionary
  olearners = [_glearnerdict[k] for k in strlearners]
  oextractors = [_gextractordict[k] for k in strextractors]
  oscalers = [_gscalersdict[k] for k in strscalers]
  autoclass.model[:olearners] = olearners
  autoclass.model[:oextractors] = oextractors
  autoclass.model[:oscalers] = oscalers

  # store pipelines
  dfpipelines = model_selection_pipeline(autoclass)
  autoclass.model[:dfpipelines] = dfpipelines

  # find the best model by evaluating the models
  modelsperf = evaluate_pipeline(autoclass, X, Y)
  sort!(modelsperf, :mean, rev=autoclass.model[:sortrev])

  # get the string name of the top model
  @show modelsperf
  bestm = filter(x -> occursin(x, modelsperf.Description[1]), keys(_glearnerdict) |> collect)[1]

  # get corresponding model object
  bestlearner = _glearnerdict[bestm]
  autoclass.model[:bestlearner] = bestlearner
  optmodel = DataFrame()
  if autoclass.model[:complexity] == "low"
    optmodel = oneblocksearch(autoclass, X, Y)
  else
    optmodel = twoblocksearch(autoclass, X, Y)
  end
  bestpipeline = optmodel.Pipeline
  # train the best pipeline and store it
  fit!(bestpipeline, X, Y)
  bestpipeline.model[:description] = optmodel.Description
  autoclass.model[:bestpipeline] = bestpipeline
  return nothing
end

function fit(clfb::AutoClassification, X::DataFrame, Y::Vector)
  autoclass = deepcopy(clfb)
  fit!(autoclass, X, Y)
  return autoclass
end

function transform!(autoclass::AutoClassification, X::DataFrame)
  bestpipeline = autoclass.model[:bestpipeline]
  transform!(bestpipeline, X)
end

function transform(autoclass::AutoClassification, X::DataFrame)
  bestpipeline = deepcopy(autoclass.model[:bestpipeline])
  transform!(bestpipeline, X)
end


end
