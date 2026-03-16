module XGBoostLearners

import PythonCall
const PYC = PythonCall

# standard included modules
using DataFrames
using Random
using ..AbsTypes
using ..Utils

import ..AbsTypes: fit, fit!, transform, transform!
export fit, fit!, transform, transform!
export XGBoostLearner, xgboostlearners

const learner_dict = Dict{String,PYC.Py}()
const XGB = PYC.pynew()

function __init__()
  PYC.pycopy!(XGB, PYC.pyimport("xgboost"))

  # Available XGBoost learners
  learner_dict["XGBClassifier"] = XGB
  learner_dict["XGBRegressor"] = XGB
  learner_dict["XGBRFClassifier"] = XGB
  learner_dict["XGBRFRegressor"] = XGB
  learner_dict["XGBRanker"] = XGB
end

"""
    XGBoostLearner(learner::String, args::Dict=Dict())

An XGBoost wrapper to load the different machine learning models.
Invoking `xgboostlearners()` will list the available learners. Please
consult XGBoost documentation for arguments to pass.

Implements `fit!` and `transform!`. 

# Arguments
- `learner::String`: Name of the XGBoost learner (e.g., "XGBClassifier", "XGBRegressor")
- `args::Dict`: Dictionary of arguments including:
  - `:name`: Custom name for the learner instance
  - `:output`: Output type (`:class` or `:numeric`)
  - `:impl_args`: Dictionary of XGBoost-specific parameters

# Examples
```julia
# Classification
xgb_clf = XGBoostLearner("XGBClassifier", 
                         Dict(:impl_args => Dict(:max_depth => 3, 
                                                :learning_rate => 0.1,
                                                :n_estimators => 100)))

# Regression
xgb_reg = XGBoostLearner("XGBRegressor"; 
                         max_depth=5, 
                         learning_rate=0.05,
                         n_estimators=200)
```
"""
mutable struct XGBoostLearner <: Learner
  name::String
  model::Dict{Symbol,Any}

  function XGBoostLearner(args=Dict{Symbol,Any}())
    default_args = Dict{Symbol,Any}(
      :name => "xgboost",
      :output => :class,
      :learner => "XGBClassifier",
      :impl_args => Dict{Symbol,Any}()
    )
    cargs = nested_dict_merge(default_args, args)
    cargs[:name] = cargs[:name] * "_" * randstring(3)
    xgb = cargs[:learner]
    if !(xgb in keys(learner_dict))
      println("$xgb is not supported.")
      println()
      xgboostlearners()
      error("Argument keyword error")
    end
    new(cargs[:name], cargs)
  end
end

function XGBoostLearner(learner::String, args::Dict)
  XGBoostLearner(Dict(:learner => learner, :name => learner, args...))
end

function XGBoostLearner(learner::String; args...)
  XGBoostLearner(Dict(:learner => learner, :name => learner, :impl_args => Dict(pairs(args))))
end

function (xgb::XGBoostLearner)(; objargs...)
  xgb.model[:impl_args] = Dict(pairs(objargs))
  xgbname = xgb.model[:learner]
  xgbobj = getproperty(learner_dict[xgbname], xgbname)
  newxgbobj = xgbobj(; objargs...)
  xgb.model[:xgboost] = newxgbobj
  return xgb
end

"""
    function xgboostlearners()

List the available XGBoost machine learners.
"""
function xgboostlearners()
  learners = keys(learner_dict) |> collect |> x -> sort(x, lt=(x, y) -> lowercase(x) < lowercase(y))
  println("syntax: XGBoostLearner(name::String, args::Dict=Dict())")
  println("where 'name' can be one of:")
  println()
  [print(learner, " ") for learner in learners]
  println()
  println()
  println("and 'args' are the corresponding learner's initial parameters.")
  println("Note: Consult XGBoost's online help for more details about the learner's arguments.")
  println()
  println("Common parameters:")
  println("  - max_depth: Maximum tree depth (default: 6)")
  println("  - learning_rate (eta): Step size shrinkage (default: 0.3)")
  println("  - n_estimators: Number of boosting rounds (default: 100)")
  println("  - objective: Learning objective (auto-detected based on learner type)")
  println("  - booster: Which booster to use ('gbtree', 'gblinear', 'dart')")
  println("  - subsample: Subsample ratio of training instances (default: 1)")
  println("  - colsample_bytree: Subsample ratio of columns (default: 1)")
  println("  - reg_alpha: L1 regularization term (default: 0)")
  println("  - reg_lambda: L2 regularization term (default: 1)")
end

function fit!(xgb::XGBoostLearner, xx::DataFrame, yy::Vector)::Nothing
  # normalize inputs
  x = xx |> Array
  y = yy
  xgb.model[:predtype] = :numeric
  if !(eltype(yy) <: Real)
    y = yy |> Vector{String}
    xgb.model[:predtype] = :alpha
  end

  impl_args = copy(xgb.model[:impl_args])
  learner = xgb.model[:learner]
  py_learner = getproperty(learner_dict[learner], learner)

  # Train
  modelobj = py_learner(; impl_args...)
  modelobj.fit(x, y)
  xgb.model[:xgboost] = modelobj
  xgb.model[:impl_args] = impl_args
  return nothing
end

function fit(xgb::XGBoostLearner, xx::DataFrame, y::Vector)::XGBoostLearner
  fit!(xgb, xx, y)
  return deepcopy(xgb)
end

function transform!(xgb::XGBoostLearner, xx::DataFrame)::Vector
  x = deepcopy(xx) |> Array
  xgboost = xgb.model[:xgboost]
  res = xgboost.predict(x)
  if xgb.model[:predtype] == :numeric
    predn = PYC.pyconvert(Vector{Float64}, res)
    return predn
  else
    predc = PYC.pyconvert(Vector{String}, res)
    return predc
  end
end

transform(xgb::XGBoostLearner, xx::DataFrame)::Vector = transform!(xgb, xx)

end

# Made with Bob
