module SKCrossValidators

using PyCall

# standard included modules
using DataFrames
using Random
using AutoMLPipeline.AbsTypes
using AutoMLPipeline.Utils

import AutoMLPipeline.CrossValidators.crossvalidate
export crossvalidate

function __init__()
    global SKM = pyimport("sklearn.metrics")
    global metric_dict = Dict(
          "roc_auc_score" => SKM.roc_auc_score,
          "accuracy_score" => SKM.accuracy_score,
          "auc" => SKM.auc,
          "average_precision_score" => SKM.average_precision_score,
          "balanced_accuracy_score" => SKM.balanced_accuracy_score,
          "brier_score_loss" => SKM.brier_score_loss,
          "classification_report" => SKM.classification_report,
          "cohen_kappa_score" => SKM.cohen_kappa_score,
          "confusion_matrix" => SKM.confusion_matrix,
          "f1_score" => SKM.f1_score,
          "fbeta_score" => SKM.fbeta_score,
          "hamming_loss" => SKM.hamming_loss,
          "hinge_loss" => SKM.hinge_loss,
          "log_loss" => SKM.log_loss,
          "matthews_corrcoef" => SKM.matthews_corrcoef,
          "multilabel_confusion_matrix" => SKM.multilabel_confusion_matrix,
          "precision_recall_curve" => SKM.precision_recall_curve,
          "precision_recall_fscore_support" => SKM.precision_recall_fscore_support,
          "precision_score" => SKM.precision_score,
          "recall_score" => SKM.recall_score,
          "roc_auc_score" => SKM.roc_auc_score,
          "roc_curve" => SKM.roc_curve,
          "jaccard_score" => SKM.jaccard_score,
          "zero_one_loss" => SKM.zero_one_loss
         )
end

function crossvalidate(pl::Machine,X::DataFrame,Y::Vector,
                       sfunc::String,nfolds=10)
    @assert sfunc in keys(metric_dict)
    pfunc = metric_dict[sfunc]
    metric(X,Y) = pfunc(X,Y)
    crossvalidate(pl,X,Y,metric,nfolds)
end

function crossvalidate(pl::Machine,X::DataFrame,Y::Vector,
                       sfunc::String,averagetype::String,nfolds=10)
    @assert sfunc in keys(metric_dict)
    pfunc = metric_dict[sfunc]
    metric(X,Y) = pfunc(X,Y,average=averagetype)
    crossvalidate(pl,X,Y,metric,nfolds)
end


end
