using Distributed
using AutoMLPipeline
using DataFrames
using Serialization
using AutoAI

nprocs() == 1 && addprocs()
@everywhere using AutoAI

dataset = getiris()
X = dataset[:, 1:4]
Y = dataset[:, 5] |> collect
#X = dataset[:, 2:end-1]

model = clsearch(X, Y; classification=true)
serialize("clmodel.bin", model)
dclmodel = deserialize("clmodel.bin")
AutoMLPipeline.transform(dclmodel, X)

X = dataset[:, 2:end-1]
Y = dataset[:, 1] |> collect

regmodel = clsearch(X, Y; classification=false)
serialize("regmodel.bin", regmodel)
dregmodel = deserialize("regmodel.bin")
AutoMLPipeline.transform(dregmodel, X)
