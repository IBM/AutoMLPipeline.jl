module AutoMLPipeline
using Reexport
using DataFrames: DataFrame

using AMLPipelineBase
using AMLPipelineBase.AbsTypes
export fit, fit!, transform, transform!, fit_transform, fit_transform!
export fit_transform_trace

using AMLPipelineBase
using AMLPipelineBase: AbsTypes, Utils, BaselineModels, Pipelines
using AMLPipelineBase: BaseFilters, FeatureSelectors, DecisionTreeLearners
using AMLPipelineBase: EnsembleMethods, CrossValidators
using AMLPipelineBase: NARemovers

export Machine, Learner, Transformer, Workflow, Computer
export holdout, kfold, score, infer_eltype, nested_dict_to_tuples,
    nested_dict_set!, nested_dict_merge, create_transformer,
    mergedict, getiris, getprofb,
    skipmean, skipmedian, skipstd,
    aggregatorclskipmissing,
    find_catnum_columns,
    train_test_split


export Baseline, Identity
export Imputer, OneHotEncoder, Wrapper
export PrunedTree, RandomForest, Adaboost
export VoteEnsemble, StackEnsemble, BestLearner
export FeatureSelector, CatFeatureSelector, NumFeatureSelector, CatNumDiscriminator
export crossvalidate
export NARemover
export @pipeline, @pipelinex
export +, |>, *, |, >>
export Pipeline, ComboPipeline

import AMLPipelineBase.AbsTypes: fit!, transform!


@reexport using OpenTelemetry
@reexport using Term
@reexport using Logging

export enableotlp
function enableotlp()
    ENV["OTEL"] = "true"
    global_tracer_provider(TracerProvider(;
        span_processor=SimpleSpanProcessor(OtlpHttpTracesExporter())
    ))
    global_logger(OtelSimpleLogger(exporter=OtlpHttpLogsExporter()))
    global_meter_provider(MeterProvider())
    MetricReader(OtlpHttpMetricsExporter())
    return nothing
end

export isotlpenabled
function isotlpenabled()
    if "OTEL" in keys(ENV)
        if ENV["OTEL"] == "true"
            return true
        end
    end
    return false
end


# --------------------------------------------

include("skpreprocessor.jl")
using .SKPreprocessors
export SKPreprocessor, skpreprocessors

include("sklearners.jl")
using .SKLearners
export SKLearner, sklearners

include("skcrossvalidator.jl")
using .SKCrossValidators
export crossvalidate
export testmacro

export skoperator
function skoperator(name::String; args...)::Machine
    sklr = keys(SKLearners.learner_dict)
    skpr = keys(SKPreprocessors.preprocessor_dict)
    if name ∈ sklr
        obj = SKLearner(name; args...)
    elseif name ∈ skpr
        obj = SKPreprocessor(name; args...)
    else
        skoperator()
        throw(ArgumentError("$name does not exist"))
    end
    return obj
end

function skoperator()
    sklr = keys(SKLearners.learner_dict)
    skpr = keys(SKPreprocessors.preprocessor_dict)
    println("Please choose among these pipeline elements:")
    println([sklr..., skpr...])
end

function fit_transform_trace(mc::Workflow, input::DataFrame=DataFrame(),
    output::Vector=Vector())::Union{Vector,DataFrame}
    if isotlpenabled()
        with_span("fit_transform $(mc.name)") do
            return fit_transform!(mc, input, output)
        end
    else
        return fit_transform!(mc, input, output)
    end
end


end # module
