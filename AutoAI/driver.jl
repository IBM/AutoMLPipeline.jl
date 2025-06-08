using AutoAI
using CSV
using DataFrames: DataFrame, nrow
using Serialization
using Distributed

nprocs() == 1 && addprocs()
@everywhere using AutoAI

df = CSV.File("./iris.csv") |> DataFrame

# classification
X = df[:, 1:end-1]
Y = df[:, end] |> collect
autoclass = AutoClassification()
fit_transform!(autoclass, X, Y)

serialize("autoclass.bin", autoclass)
bestautoclass = deserialize("./autoclass.bin")
transform!(bestautoclass, X)

# regression

learners = ["rfr", "gbr", "ridger",
    "svr", "adar", "dtr", "lassor",
    "enetr", "ardr", "larsr", "sgdr",
    "kridger"
]

scalers = ["rb", "pt",
    "norm", "mx",
    "std", "noop"]

extractors = ["pca", "fa",
    "ica", "noop"]

ndx = sample(1:nrow(df), nrow(df); replace=false)

X = df[ndx[1:100], [1, 2, 3, 5]]
Y = df[ndx[1:100], 4] |> collect
autoreg = AutoRegression(Dict(:learners => learners,
    :extractors => extractors,
    :scalers => scalers))
YY = fit_transform!(autoreg, X, Y)

YY = transform!(autoreg, df[ndx[101:150], :])

plot(1:length(YY), [YY, df[ndx[101:150], 4]])

plot(YY)
plot(df[ndx[101:150], 4])

serialize("autoreg.bin", autoreg)
bestautoreg = deserialize("./autoreg.bin")
transform!(bestautoreg, X)
