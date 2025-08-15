module AutoRegressions
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
export AutoRegression

include("./pipelinesearch.jl")

# define learners
const rfr = SKLearner("RandomForestRegressor", Dict(:name => "rfr"))
const adar = SKLearner("AdaBoostRegressor", Dict(:name => "adar"))
const gbr = SKLearner("GradientBoostingRegressor", Dict(:name => "gbr"))
const ridger = SKLearner("Ridge", Dict(:name => "ridger"))
const svr = SKLearner("SVR", Dict(:name => "svr"))
const dtr = SKLearner("DecisionTreeRegressor", Dict(:name => "dtr"))
const lassor = SKLearner("Lasso", Dict(:name => "lassor"))
const enetr = SKLearner("ElasticNet", Dict(:name => "enetr"))
const ardr = SKLearner("ARDRegression", Dict(:name => "ardr"))
const larsr = SKLearner("Lars", Dict(:name => "larsr"))
const sgdr = SKLearner("SGDRegressor", Dict(:name => "sgdr"))
const kridger = SKLearner("KernelRidge", Dict(:name => "kridger"))

const _glearnerdictr = Dict("rfr" => rfr, "gbr" => gbr, "ridger" => ridger,
  "svr" => svr, "adar" => adar, "dtr" => dtr, "lassor" => lassor,
  "enetr" => enetr, "ardr" => ardr, "larsr" => larsr, "sgdr" => sgdr,
  "kridger" => kridger
)

# define customized type
mutable struct AutoRegression <: Workflow
  name::String
  model::Dict{Symbol,Any}

  function AutoRegression(args=Dict())
    default_args = Dict(
      :name => "regsearchblock",
      :complexity => "low",
      :prediction_type => "regression",
      :nfolds => 3,
      :metric => "mean_squared_error",
      :nworkers => 5,
      :learners => ["rfr", "svr", "gbr", "adar"],
      :scalers => ["norm", "pt", "mx", "std", "rb", "pt", "noop"],
      :extractors => ["pca", "ica", "fa", "noop"],
      :sortrev => false,
      :impl_args => Dict()
    )
    cargs = nested_dict_merge(default_args, args)
    cargs[:name] = cargs[:name] * "_" * randstring(3)
    learners = cargs[:learners]
    for learner in learners
      if !(learner in keys(_glearnerdictr))
        println("$learner is not supported.")
        println()
        listreglearners()
        error("Argument keyword error")
      end
    end
    new(cargs[:name], cargs)
  end
end

function AutoRegression(learners::Vector{String}, args::Dict)
  AutoRegression(Dict(:learners => learners, args...))
end

function AutoRegression(learners::Vector{String}; args...)
  AutoRegression(Dict(:learners => learners, :impl_args => Dict(pairs(args))))
end


function listreglearners()
  println("Use available learners:")
  [print(learner, " ") for learner in keys(_glearnerdictr)]
  println()
end

function fit!(autoreg::AutoRegression, X::DataFrame, Y::Vector)
  strscalers = autoreg.model[:scalers]
  strextractors = autoreg.model[:extractors]
  strlearners = autoreg.model[:learners]

  # get objects from dictionary
  olearners = [_glearnerdictr[k] for k in strlearners]
  oextractors = [_gextractordict[k] for k in strextractors]
  oscalers = [_gscalersdict[k] for k in strscalers]
  autoreg.model[:olearners] = olearners
  autoreg.model[:oextractors] = oextractors
  autoreg.model[:oscalers] = oscalers

  # store pipelines
  dfpipelines = model_selection_pipeline(autoreg)
  autoreg.model[:dfpipelines] = dfpipelines

  # find the best model by evaluating the models
  modelsperf = evaluate_pipeline(autoreg, X, Y)
  sort!(modelsperf, :mean, rev=autoreg.model[:sortrev])

  # get the string name of the top model
  @show modelsperf
  bestm = filter(x -> occursin(x, modelsperf.Description[1]), keys(_glearnerdictr) |> collect)[1]

  # get corresponding model object
  bestlearner = _glearnerdictr[bestm]
  autoreg.model[:bestlearner] = bestlearner
  optmodel = DataFrame()
  if autoreg.model[:complexity] == "low"
    optmodel = oneblocksearch(autoreg, X, Y)
  else
    optmodel = twoblocksearch(autoreg, X, Y)
  end
  bestpipeline = optmodel.Pipeline
  # train the best pipeline and store it
  fit!(bestpipeline, X, Y)
  bestpipeline.model[:description] = optmodel.Description
  autoreg.model[:bestpipeline] = bestpipeline
  return nothing
end

function fit(regb::AutoRegression, X::DataFrame, Y::Vector)
  autoreg = deepcopy(regb)
  fit!(autoreg, X, Y)
  return autoreg
end

function transform!(autoreg::AutoRegression, X::DataFrame)
  bestpipeline = autoreg.model[:bestpipeline]
  transform!(bestpipeline, X)
end

function transform(autoreg::AutoRegression, X::DataFrame)
  bestpipeline = deepcopy(autoreg.model[:bestpipeline])
  transform!(bestpipeline, X)
end

end
