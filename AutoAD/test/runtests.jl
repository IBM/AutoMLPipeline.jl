module TestAutoAD
using Distributed

nprocs() == 1 && addprocs()
@everywhere using AutoAD

@everywhere include("./test_caret_anomalydetector.jl")
include("./test_skanomalydetector.jl")
@everywhere include("./test_caret_tspredictor.jl")
end
