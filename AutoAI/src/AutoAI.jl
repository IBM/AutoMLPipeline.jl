module AutoAI
__precompile__(false)

using Reexport
@reexport using AutoMLPipeline

using AutoMLPipeline
using DataFrames
using CSV
using Random

#nprocs() == 1 && addprocs(exeflags=["--project=$(Base.active_project())"])

Random.seed!(10)

include("twoblocks.jl")
using .TwoBlocksPipelines

export clsearch
function clsearch(X::DataFrame, Y::Vector; classification=true)
    resultsdf = bestmodelsearch(X, Y; classification)
    println("best pipeline is: ", resultsdf[1, 1], ", with mean-sd ", resultsdf[1, 2], " ± ", resultsdf[1, 3])
    bestmodel = resultsdf[1, :Pipeline]
    fit!(bestmodel, X, Y)
    return bestmodel
end

end # module AutoAI
