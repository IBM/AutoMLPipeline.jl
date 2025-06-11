module TestAutoML
using Test
using CSV
using AutoAI
using DataFrames: DataFrame
using Serialization
using Distributed
using Statistics

nprocs() == 1 && addprocs()
@everywhere using AutoAI

df = CSV.File("./iris.csv") |> DataFrame

# classification
function classify(df)
    X = df[:, 1:end-1]
    Y = df[:, end] |> collect
    autoclass = AutoClassification()
    Yhat = fit_transform!(autoclass, X, Y)
    return Yhat, Y
end

# regression
function curvefit(df)
    X = df[:, [1, 2, 3, 5]]
    Y = df[:, 4] |> collect
    autoreg = AutoRegression()
    Yhat = fit_transform!(autoreg, X, Y)
    return Yhat, Y
end

@testset "autoclassification and autoregression" begin
    Yhatc, Yc = classify(df)
    Yhatr, Yr = curvefit(df)
    @test mean(Yhatc .== Yc) > 0.95
    @test mean((Yhatr .- Yr) .^ 2) < 0.03
end

end

