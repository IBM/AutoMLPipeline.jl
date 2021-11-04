using Distributed
using DataFrames
using CSV


nprocs() == 1 && addprocs(; exeflags = "--project")
@everywhere include("twoblocks.jl")
@everywhere using Main.TwoBlocksPipeline

#dataset = getiris()
#X = dataset[:,1:4]
#Y = dataset[:,5] |> collect
#X = dataset[:,2:end-1]

dataset=CSV.read("environment_data.csv",DataFrame)
X = select(dataset, Not([:state,:time]))
Y = dataset.state

results=TwoBlocksPipeline.twoblockspipelinesearch(X,Y)
println("best pipeline is ", results[1]," with mean-sd ",results[2], " ± ",results[3])
