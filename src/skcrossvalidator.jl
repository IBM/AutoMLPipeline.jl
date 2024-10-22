module SKCrossValidators

import PythonCall
const PYC = PythonCall

# standard included modules
using DataFrames
using Random
using ..AbsTypes
using ..Utils
using AutoMLPipeline: isotlpenabled

using OpenTelemetry
using Term
using Logging

import ..CrossValidators: crossvalidate
export crossvalidate
export testmacro

const metric_dict = Dict{String,PYC.Py}()
const SKM = PYC.pynew()

function __init__()
    PYC.pycopy!(SKM, PYC.pyimport("sklearn.metrics"))

    metric_dict["roc_auc_score"] = SKM.roc_auc_score
    metric_dict["accuracy_score"] = SKM.accuracy_score
    metric_dict["auc"] = SKM.auc
    metric_dict["average_precision_score"] = SKM.average_precision_score
    metric_dict["balanced_accuracy_score"] = SKM.balanced_accuracy_score
    metric_dict["brier_score_loss"] = SKM.brier_score_loss
    metric_dict["classification_report"] = SKM.classification_report
    metric_dict["cohen_kappa_score"] = SKM.cohen_kappa_score
    metric_dict["confusion_matrix"] = SKM.confusion_matrix
    metric_dict["f1_score"] = SKM.f1_score
    metric_dict["fbeta_score"] = SKM.fbeta_score
    metric_dict["hamming_loss"] = SKM.hamming_loss
    metric_dict["hinge_loss"] = SKM.hinge_loss
    metric_dict["log_loss"] = SKM.log_loss
    metric_dict["matthews_corrcoef"] = SKM.matthews_corrcoef
    metric_dict["multilabel_confusion_matrix"] = SKM.multilabel_confusion_matrix
    metric_dict["precision_recall_curve"] = SKM.precision_recall_curve
    metric_dict["precision_recall_fscore_support"] = SKM.precision_recall_fscore_support
    metric_dict["precision_score"] = SKM.precision_score
    metric_dict["recall_score"] = SKM.recall_score
    metric_dict["roc_auc_score"] = SKM.roc_auc_score
    metric_dict["roc_curve"] = SKM.roc_curve
    metric_dict["jaccard_score"] = SKM.jaccard_score
    metric_dict["zero_one_loss"] = SKM.zero_one_loss
    # regression
    metric_dict["mean_squared_error"] = SKM.mean_squared_error
    metric_dict["mean_squared_log_error"] = SKM.mean_squared_log_error
    metric_dict["mean_absolute_error"] = SKM.mean_absolute_error
    metric_dict["median_absolute_error"] = SKM.median_absolute_error
    metric_dict["r2_score"] = SKM.r2_score
    metric_dict["max_error"] = SKM.max_error
    metric_dict["mean_poisson_deviance"] = SKM.mean_poisson_deviance
    metric_dict["mean_gamma_deviance"] = SKM.mean_gamma_deviance
    metric_dict["mean_tweedie_deviance"] = SKM.mean_tweedie_deviance
    metric_dict["explained_variance_score"] = SKM.explained_variance_score
end

function checkfun(sfunc::String)
    if !(sfunc in keys(metric_dict))
        println("$sfunc metric is not supported")
        println("metric: ", keys(metric_dict))
        error("Metric keyword error")
    end
end

"""
    crossvalidate(pl::Machine,X::DataFrame,Y::Vector,sfunc::String="balanced_accuracy_score",nfolds=10)

Runs K-fold cross-validation using balanced accuracy as the default. It support the 
following metrics for classification:
- "accuracy_score"
- "balanced_accuracy_score"
- "cohen_kappa_score"
- "jaccard_score"
- "matthews_corrcoef"
- "hamming_loss"
- "zero_one_loss"
- "f1_score"
- "precision_score"
- "recall_score"

and the following metrics for regression:
- "mean_squared_error"
- "mean_squared_log_error"
- "median_absolute_error"
- "r2_score"
- "max_error"
- "explained_variance_score"
"""
function crossvalidate(pl::Machine, X::DataFrame, Y::Vector,
    sfunc::String; nfolds=10, verbose::Bool=true)
    function _crossvalidate(pl::Machine, X::DataFrame, Y::Vector,
        sfunc::String; nfolds=10, verbose::Bool=true)
        YC = Y
        if !(eltype(YC) <: Real)
            YC = Y |> Vector{String}
        end
        checkfun(sfunc)
        pfunc = metric_dict[sfunc]
        metric(a, b) = pfunc(a, b) |> (x -> PYC.pyconvert(Float64, x))
        res = crossvalidate(pl, X, YC, metric, nfolds, verbose)
        return res
    end
    if isotlpenabled()
        with_span("crossvalidate $(pl.name) $sfunc") do
            return _crossvalidate(pl, X, Y, sfunc; nfolds, verbose)
        end
    else
        return _crossvalidate(pl, X, Y, sfunc; nfolds, verbose)
    end
end

function crossvalidate(pl::Machine, X::DataFrame, Y::Vector, sfunc::String, nfolds::Int)
    crossvalidate(pl, X, Y, sfunc; nfolds)
end

function crossvalidate(pl::Machine, X::DataFrame, Y::Vector, sfunc::String, verbose::Bool)
    crossvalidate(pl, X, Y, sfunc; verbose)
end

function crossvalidate(pl::Machine, X::DataFrame, Y::Vector,
    sfunc::String, nfolds::Int, verbose::Bool)
    crossvalidate(pl, X, Y, sfunc; nfolds, verbose)
end

function crossvalidate(pl::Machine, X::DataFrame, Y::Vector,
    sfunc::String, averagetype::String; nfolds=10, verbose::Bool=true)
    function _crossvalidate(pl::Machine, X::DataFrame, Y::Vector,
        sfunc::String, averagetype::String; nfolds=10, verbose::Bool=true)
        checkfun(sfunc)
        pfunc = metric_dict[sfunc]
        metric(a, b) = pfunc(a, b, average=averagetype) |> (x -> PYC.pyconvert(Float64, x))
        res = crossvalidate(pl, X, Y, metric, nfolds, verbose)
        return res
    end
    if isotlpenabled()
        with_span("crossvalidate $(pl.name) $sfunc") do
            return _crossvalidate(pl, X, Y, sfunc,averagetype; nfolds, verbose)
        end
    else
        return _crossvalidate(pl, X, Y, sfunc,averagetype; nfolds, verbose)
    end
end

function testmacro()
    if isotlpenabled()
        println("ok")
    else
        println("no ok")
    end
end

end
