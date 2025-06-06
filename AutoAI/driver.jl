using AutoAI
using CSV
using DataFrames: DataFrame
using Serialization
using Distributed

nprocs() == 1 && addprocs()
@everywhere using AutoAI

df = CSV.File("./iris.csv") |> DataFrame
X = df[:, 1:end-1]
Y = df[:, end] |> collect
clf = ClfSearchBlock()
fit_transform!(clf, X, Y)

serialize("clfblock.bin", clf)
bestm = deserialize("./clfblock.bin")

transform!(bestm, X)
