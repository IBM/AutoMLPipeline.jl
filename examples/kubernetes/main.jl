using Distributed, K8sClusterManagers
using DataFrames
using CSV


workers = parse(Int,ARGS[1])
cpu_fraction = parse(Int,ARGS[2])
addprocs(K8sClusterManager(workers,cpu=cpu_fraction); exeflags="--project")

#addprocs(K8sClusterManager(10, cpu=1, memory="300Mi", pending_timeout=300); exeflags="--project")
#nprocs() == 1 && addprocs(; exeflags = "--project")

@everywhere include("twoblocks.jl")
@everywhere using Main.TwoBlocksPipeline

dataset = getiris()
X = dataset[:,1:4]
Y = dataset[:,5] |> collect
X = dataset[:,2:end-1]

#dataset=CSV.read("environment_data.csv",DataFrame)
#X = select(dataset, Not([:state,:time]))
#Y = dataset.state

results=TwoBlocksPipeline.twoblockspipelinesearch(X,Y)
println(results)
println("best pipeline is: ", results[1,1],", with mean-sd ",results[1,2], " ± ",results[1,3])
