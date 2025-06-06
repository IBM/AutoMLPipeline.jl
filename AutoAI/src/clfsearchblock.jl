module ClfSearchBlocks
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
export ClfSearchBlock

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
mutable struct ClfSearchBlock <: Workflow
    name::String
    model::Dict{Symbol,Any}

    function ClfSearchBlock(args=Dict())
        default_args = Dict(
            :name => "clfsearchblock",
            :complexity => "high",
            :prediction_type => "classification",
            :nfolds => 3,
            :metric => "balanced_accuracy_score",
            :nworkers => 5,
            :learners => ["rfc", "rbfsvc", "gbc"],
            :scalers => ["norm", "pt"],
            :extractors => ["pca"],
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

function listclasslearners()
    println("Use available learners:")
    [print(learner, " ") for learner in keys(_glearnerdict)]
    println()
end

function fit!(clfblock::ClfSearchBlock, X::DataFrame, Y::Vector)
    strscalers = clfblock.model[:scalers]
    strextractors = clfblock.model[:extractors]
    strlearners = clfblock.model[:learners]

    # get objects from dictionary
    olearners = [_glearnerdict[k] for k in strlearners]
    oextractors = [_gextractordict[k] for k in strextractors]
    oscalers = [_gscalersdict[k] for k in strscalers]
    clfblock.model[:olearners] = olearners
    clfblock.model[:oextractors] = oextractors
    clfblock.model[:oscalers] = oscalers

    # store pipelines
    dfpipelines = model_selection_pipeline(clfblock)
    clfblock.model[:dfpipelines] = dfpipelines

    # find the best model by evaluating the models
    modelsperf = evaluate_pipeline(clfblock, X, Y)
    sort!(modelsperf, :mean, rev=clfblock.model[:sortrev])

    # get the string name of the top model
    @show modelsperf
    bestm = filter(x -> occursin(x, modelsperf.Description[1]), keys(_glearnerdict) |> collect)[1]

    # get corresponding model object
    bestlearner = _glearnerdict[bestm]
    clfblock.model[:bestlearner] = bestlearner
    optmodel = DataFrame()
    if clfblock.model[:complexity] == "low"
        optmodel = oneblocksearch(clfblock, X, Y)
    else
        optmodel = twoblocksearch(clfblock, X, Y)
    end
    bestpipeline = optmodel.Pipeline
    # train the best pipeline and store it
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
