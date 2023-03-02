# make sure local environment is activated
using Pkg
Pkg.activate(".")

using Distributed
using DataFrames
using CSV
using Random

# add workers
nprocs() ==1 && addprocs(exeflags=["--project=$(Base.active_project())"])
workers()

# disable warnings
@everywhere import PythonCall
@everywhere const PYC=PythonCall
@everywhere warnings = PYC.pyimport("warnings")
@everywhere warnings.filterwarnings("ignore")

Random.seed!(10)
@everywhere include("twoblocks.jl")
@everywhere using Main.TwoBlocksPipeline

dataset = getiris()
X = dataset[:,1:4]
Y = dataset[:,5] |> collect
X = dataset[:,2:end-1]

results=TwoBlocksPipeline.twoblockspipelinesearch(X,Y)
println("best pipeline is: ", results[1,1],", with mean-sd ",results[1,2], " ± ",results[1,3])
