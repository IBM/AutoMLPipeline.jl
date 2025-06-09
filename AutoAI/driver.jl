using AutoAI
using CSV
using DataFrames: DataFrame
using Serialization
using Distributed

nprocs() == 1 && addprocs()
@everywhere using AutoAI

df = CSV.File("./test/iris.csv") |> DataFrame

X = df[:, 1:end-1]
Y = df[:, end] |> collect
autoclass = AutoClassification()
fit_transform!(autoclas, X, Y)

serialize("autoclass.bin", autoclass)
bestm = deserialize("./autoclass.bin")
transform!(bestm, X)

X = df[:, [1, 2, 3, 5]]
Y = df[:, 4] |> collect
autoreg = AutoRegression()
Yhat = fit_transform!(autoreg, X, Y)

serialize("autoreg.bin", autoreg)
bestm = deserialize("./autoreg.bin")
transform!(bestm, X)
