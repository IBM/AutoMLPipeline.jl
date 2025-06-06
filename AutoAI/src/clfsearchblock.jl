module ClfSearchBlocks
# classification search blocks

export twoblocksearch, oneblocksearch

using Distributed
using AutoMLPipeline
using DataFrames: DataFrame
using AutoMLPipeline: score
using Random
using ..AbsTypes
using ..Utils

import ..AbsTypes: fit, fit!, transform, transform!
export fit, fit!, transform, transform!
export ClfSearchBlock

# disable truncation of dataframes columns
#import Base.show
#show(df::AbstractDataFrame) = show(df, truncate=0)
#show(io::IO, df::AbstractDataFrame) = show(io, df; truncate=0)

# define scalers
const rb = SKPreprocessor("RobustScaler", Dict(:name => "rb"))
const pt = SKPreprocessor("PowerTransformer", Dict(:name => "pt"))
const norm = SKPreprocessor("Normalizer", Dict(:name => "norm"))
const mx = SKPreprocessor("MinMaxScaler", Dict(:name => "mx"))
const std = SKPreprocessor("StandardScaler", Dict(:name => "std"))
# define extractors
const pca = SKPreprocessor("PCA", Dict(:name => "pca"))
const fa = SKPreprocessor("FactorAnalysis", Dict(:name => "fa"))
const ica = SKPreprocessor("FastICA", Dict(:name => "ica"))
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
# preprocessing
const noop = Identity(Dict(:name => "noop"))
const ohe = OneHotEncoder(Dict(:name => "ohe"))
const catf = CatFeatureSelector(Dict(:name => "catf"))
const numf = NumFeatureSelector(Dict(:name => "numf"))

const vscalers = [rb, pt, norm, mx, std, noop]
const _gscalersdict = Dict("rb" => rb, "pt" => pt,
    "norm" => norm, "mx" => mx,
    "std" => std, "noop" => noop)
const vextractors = [pca, fa, ica, noop]
const _gextractordict = Dict("pca" => pca, "fa" => fa,
    "ica" => ica, "noop" => noop)
const vlearners = [rfc, gbc, lsvc, rbfsvc, adac, dtc,
    etc, ridgec, sgdc, bgc, pac]
const _glearnerdict = Dict("rfc" => rfc, "gbc" => gbc,
    "lsvc" => lsvc, "rbfsvc" => rbfsvc, "adac" => adac,
    "dtc" => dtc, "etc" => etc, "ridgec" => ridgec,
    "sgdc" => sgdc, "bgc" => bgc, "pac" => pac
)

# define customized type
mutable struct ClfSearchBlock <: Workflow
    name::String
    model::Dict{Symbol,Any}

    function ClfSearchBlock(args=Dict())
        default_args = Dict(
            :name => "clfsearchblock",
            :complexity => "high",
            :prediction_type => "classification",
            :nfolds => 3,
            :nworkers => 5,
            :data_path => "iris.csv",
            :learners => ["rfc", "rbfsvc", "gbc"],
            :scalers => ["norm", "pt"],
            :extractors => ["pca"],
            :impl_args => Dict()
        )
        cargs = nested_dict_merge(default_args, args)
        cargs[:name] = cargs[:name] * "_" * randstring(3)
        learners = cargs[:learners]
        for learner in learners
            if !(learner in keys(_glearnerdict))
                println("$learner is not supported.")
                println()
                listlearners()
                error("Argument keyword error")
            end
        end
        new(cargs[:name], cargs)
    end
end

function listlearners()
    println("Use available learners:")
    [print(learner, " ") for learner in keys(_glearnerdict)]
    println()
end


function oneblock_pipeline_factory(scalers, extractors, learners)
    results = @distributed (vcat) for lr in learners
        @distributed (vcat) for xt in extractors
            @distributed (vcat) for sc in scalers
                # baseline preprocessing
                prep = @pipeline ((catf |> ohe) + numf)
                # one-block prp
                expx = @pipeline prep |> (sc |> xt) |> lr
                scn = sc.name[1:end-4]
                xtn = xt.name[1:end-4]
                lrn = lr.name[1:end-4]
                pname = "($scn |> $xtn) |> $lrn"
                DataFrame(Description=pname, Pipeline=expx)
            end
        end
    end
    return results
end

function evaluate_pipeline(dfpipelines, X, Y; folds=3)
    res = @distributed (vcat) for prow in eachrow(dfpipelines)
        perf = crossvalidate(prow.Pipeline, X, Y, "balanced_accuracy_score"; nfolds=folds)
        DataFrame(; Description=prow.Description, mean=perf.mean, sd=perf.std, prow.Pipeline)
    end
    return res
end

function twoblock_pipeline_factory(scalers, extractors, learners)
    results = @distributed (vcat) for lr in learners
        @distributed (vcat) for xt1 in extractors
            @distributed (vcat) for xt2 in extractors
                @distributed (vcat) for sc1 in scalers
                    @distributed (vcat) for sc2 in scalers
                        prep = @pipeline ((catf |> ohe) + numf)
                        expx = @pipeline prep |> ((sc1 |> xt1) + (sc2 |> xt2)) |> lr
                        scn1 = sc1.name[1:end-4]
                        xtn1 = xt1.name[1:end-4]
                        scn2 = sc2.name[1:end-4]
                        xtn2 = xt2.name[1:end-4]
                        lrn = lr.name[1:end-4]
                        pname = "($scn1 |> $xtn1) + ($scn2 |> $xtn2) |> $lrn"
                        DataFrame(Description=pname, Pipeline=expx)
                    end
                end
            end
        end
    end
    return results
end

function model_selection_pipeline(learners)
    results = @distributed (vcat) for lr in learners
        prep = @pipeline ((catf |> ohe) + numf)
        expx = @pipeline prep |> (rb |> pca) |> lr
        pname = "(rb |> pca) |> $(lr.name[1:end-4])"
        DataFrame(Description=pname, Pipeline=expx)
    end
    return results
end

function lname(n::Learner)
    n.name[1:end-4]
end

function twoblocksearch(X, Y, oscalers, oextractors, bestmodel, nfolds)
    # use the best model to generate pipeline search
    dfp = twoblock_pipeline_factory(oscalers, oextractors, [bestmodel])
    # evaluate the pipeline
    bestp = evaluate_pipeline(dfp, X, Y; folds=nfolds)
    sort!(bestp, :mean, rev=true)
    show(bestp; allrows=false, truncate=1, allcols=false)
    println()
    optmodel = bestp[1, :]
    return optmodel
end


function oneblocksearch(X, Y, oscalers, oextractors, bestmodel, nfolds)
    # use the best model to generate pipeline search
    dfp = oneblock_pipeline_factory(oscalers, oextractors, [bestmodel])
    ## evaluate the pipeline
    bestp = evaluate_pipeline(dfp, X, Y; folds=nfolds)
    sort!(bestp, :mean, rev=true)
    show(bestp; allrows=false, truncate=1, allcols=false)
    println()
    optmodel = bestp[1, :]
    return optmodel
end

function fit!(clfblock::ClfSearchBlock, X::DataFrame, Y::Vector)
    strscalers = clfblock.model[:scalers]
    strextractors = clfblock.model[:extractors]
    strlearners = clfblock.model[:learners]
    # get objects from dictionary
    olearners = [_glearnerdict[k] for k in strlearners]
    oextractors = [_gextractordict[k] for k in strextractors]
    oscalers = [_gscalersdict[k] for k in strscalers]

    nfolds = clfblock.model[:nfolds]
    dfpipes = model_selection_pipeline(olearners)
    # find the best model by evaluating the models
    modelsperf = evaluate_pipeline(dfpipes, X, Y; folds=nfolds)
    sort!(modelsperf, :mean, rev=true)
    # get the string name of the top model
    @show modelsperf
    bestm = filter(x -> occursin(x, modelsperf.Description[1]), keys(_glearnerdict) |> collect)[1]
    # get corresponding model object
    bestmodel = _glearnerdict[bestm]
    optmodel = DataFrame()
    if clfblock.model[:complexity] == "low"
        optmodel = oneblocksearch(X, Y, oscalers, oextractors, bestmodel, nfolds)
    else
        optmodel = twoblocksearch(X, Y, oscalers, oextractors, bestmodel, nfolds)
    end
    bestpipeline = optmodel.Pipeline
    fit!(bestpipeline, X, Y)
    bestpipeline.model[:description] = optmodel.Description
    clfblock.model[:bestpipeline] = bestpipeline
    return nothing
end

function fit(clfb::ClfSearchBlock, X::DataFrame, Y::Vector)
    clfblock = deepcopy(clfb)
    fit!(clfblock, X, Y)
    return clfblock
end

function transform!(clfblock::ClfSearchBlock, X::DataFrame)
    bestpipeline = clfblock.model[:bestpipeline]
    transform!(bestpipeline, X)
end

function transform(clfblock::ClfSearchBlock, X::DataFrame)
    bestpipeline = deepcopy(clfblock.model[:bestpipeline])
    transform!(bestpipeline, X)
end

end
