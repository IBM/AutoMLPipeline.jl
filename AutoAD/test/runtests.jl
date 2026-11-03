module TestAutoAD
using Distributed

nprocs() == 1 && addprocs()
@everywhere using AutoAD

@everywhere include("./test_caret_anomalydetector.jl")
include("./test_skanomalydetector.jl")
include("./test_outlierdetector.jl")
end
