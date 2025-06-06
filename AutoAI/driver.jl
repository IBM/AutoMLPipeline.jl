using AutoAI
using CSV
using DataFrames: DataFrame
using Serialization
using Distributed

nprocs() == 1 && addprocs()
@everywhere using AutoAI

df = CSV.File("./iris.csv") |> DataFrame

# classification
X = df[:, 1:end-1]
Y = df[:, end] |> collect
clf = ClfSearchBlock()
fit!(clf, X, Y)
fit_transform!(clf, X, Y)

# regression
X = df[:, [1, 2, 3, 5]]
Y = df[:, 4] |> collect
reg = RegSearchBlock()
fit!(reg, X, Y)
fit_transform!(reg, X, Y)

serialize("clfblock.bin", clf)
bestm = deserialize("./clfblock.bin")
