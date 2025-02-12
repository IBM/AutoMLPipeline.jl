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

export tbsearch
function tbsearch()
    greet() = print("Hello World!")

    dataset = getiris()
    X = dataset[:, 1:4]
    Y = dataset[:, 5] |> collect
    X = dataset[:, 2:end-1]

    results = TwoBlocksPipelines.twoblockspipelinesearch(X, Y)
    println("best pipeline is: ", results[1, 1], ", with mean-sd ", results[1, 2], " ± ", results[1, 3])
end

end # module AutoAI
