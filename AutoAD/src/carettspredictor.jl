module CaretTSPredictors

using Distributed
import PythonCall
const PYC = PythonCall

# standard included modules
using DataFrames: DataFrame
using Random
using ..AbsTypes
using ..Utils

import ..AbsTypes: fit, fit!, transform, transform!
export fit, fit!, transform, transform!
export CaretTSPredictor, carettspredictors
export carettsdriver

function carettspredictors()
  println("Use available learners:")
  [print(learner, " ") for learner in keys(carettspredictor_dict)]
  println()
end

const CTS = PYC.pynew()
const PD = PYC.pynew()

function __init__()
  PYC.pycopy!(CTS, PYC.pyimport("pycaret.time_series"))
  PYC.pycopy!(PD, PYC.pyimport("pandas"))
end

const carettspredictor_dict = Dict{String,PYC.Py}(
  "exp_smooth" => CTS, "ets" => CTS, "arima" => CTS,
  "auto_arima" => CTS, "theta" => CTS,
  "huber_cds_dt" => CTS, "knn_cds_dt" => CTS,
  "lr_cds_dt" => CTS, "ridge_cds_dt" => CTS, "br_cds_dt" => CTS,
  "en_cds_dt" => CTS, "lasso_cds_dt" => CTS, "et_cds_dt" => CTS,
  "rf_cds_dt" => CTS, "dt_cds_dt" => CTS, "lightgbm_cds_dt" => CTS,
  "ada_cds_dt" => CTS, "omp_cds_dt" => CTS, "gbr_cds_dt" => CTS,
  "llar_cds_dt" => CTS, "naive" => CTS,
  "polytrend" => CTS, "croston" => CTS, "grand_means" => CTS,
  "bats" => CTS, "tbats" => CTS
  #"snaive","stlf","prophet","catboost_cds_dt"

)

const carettsexp_dict = Dict{String,PYC.Py}()
carettsexp_dict["TSForecastingExperiment"] = CTS


mutable struct CaretTSPredictor <: Learner
  name::String
  model::Dict{Symbol,Any}
  function CaretTSPredictor(args=Dict())
    default_args = Dict(
      :name => "caretts",
      :verbose => false,
      :learner => "auto",
      :experiment => "TSForecastingExperiment",
      :output => "forecast",
      :forecast_horizon => 10,
      :impl_args => Dict{Symbol,Any}()
    )
    cargs = nested_dict_merge(default_args, args)
    cargs[:name] = cargs[:name] * "_" * randstring(3)
    skl = cargs[:learner]
    if skl != "auto" && !(skl in keys(carettspredictor_dict))
      println("$skl is not supported.")
      println()
      carettspredictors()
      error("Argument keyword error")
    end
    new(cargs[:name], cargs)
  end
end

function CaretTSPredictor(learner::String, args::Dict)
  CaretTSPredictor(Dict(:learner => learner, :name => learner, args...))
end

function CaretTSPredictor(learner::String; args...)
  CaretTSPredictor(Dict(:learner => learner, :name => learner, :impl_args => Dict(pairs(args))))
end

function fit!(adl::CaretTSPredictor, xx::DataFrame, ::Vector=[])::Nothing
  xh = xx |> Array
  py_dataframe = getproperty(PD, "DataFrame")
  x = py_dataframe(xh)
  impl_args = copy(adl.model[:impl_args])
  expt = adl.model[:experiment]
  learner = adl.model[:learner]
  py_experiment = getproperty(carettsexp_dict[expt], expt)()
  _verbose = adl.model[:verbose]
  py_experiment.setup(x, session_id=123, verbose=_verbose)
  if learner != "auto"
    clearner = py_experiment.create_model(learner, verbose=_verbose)
    @info "evaluating the model: $clearner"
    finalmodel = py_experiment.finalize_model(clearner)
    adl.model[:finalmodel] = finalmodel
  else
    best = py_experiment.compare_models(verbose=_verbose)
    @info "evaluating the best model: $best"
    finalmodel = py_experiment.finalize_model(best)
    adl.model[:finalmodel] = finalmodel
  end

  # save experiment
  #adl.model[:py_experiment] = py_experiment
  return nothing
end

function transform!(adl::CaretTSPredictor, xx::DataFrame)
  xh = deepcopy(xx) |> Array
  py_dataframe = getproperty(PD, "DataFrame")
  x = py_dataframe(xh)
  learner = adl.model[:learner]
  expt = adl.model[:experiment]
  py_experiment = getproperty(carettsexp_dict[expt], expt)()
  _verbose = adl.model[:verbose]
  py_experiment.setup(x, session_id=123, verbose=_verbose)
  forecast_horizon = adl.model[:forecast_horizon]
  finalmodel = adl.model[:finalmodel]
  res = py_experiment.predict_model(finalmodel, fh=forecast_horizon, verbose=_verbose)
  finalres = res.y_pred |> PYC.PyArray |> Vector
  return finalres
end

function carettsdriver()
  DT = PYC.pyimport("pycaret.datasets")
  PD = PYC.pyimport("pandas")
  get_data = getproperty(DT, "get_data")
  dftmp = get_data("airline") |> collect
  dt=PYC.pyconvert.(Float64,dftmp)
  df = DataFrame(x1=dt)
  #df = rand(100, 1) |> x -> DataFrame(x, :auto)
  #bmodel = CaretTSPredictor("auto_arima", Dict(:verbose => true,:forecast_horizon=>30))
  bmodel = CaretTSPredictor("auto", Dict(:verbose => true,:forecast_horizon=>30))
  bestres = fit_transform!(bmodel, df)
  #tabres = @sync @distributed (hcat) for learner in ["ridge_cds_dt", "auto_arima", "ets", "rf_cds_dt"]
  #  model = CaretTSPredictor(learner, Dict(:verbose => false))
  #  res = fit_transform!(model, df)
  #  DataFrame(learner => res)
  #end
  #@show hcat(tabres, DataFrame(:best => bestres))
  #print(bmodel.model[:finalmodel])
  #return nothing
  #hcat(tabres, DataFrame(:best => bestres))
  ndx1=1:length(df.x1)
  ndx2=(length(ndx1)+1):length(ndx1)+length(bestres)
  (ndx1,df.x1,ndx2,bestres)
end

end

