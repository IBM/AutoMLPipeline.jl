module TestAutoAI
using Distributed

nprocs() == 1 && addprocs()
@everywhere using AutoAI

@everywhere include("./test_automl.jl")
@everywhere include("./test_caret_anomalydetector.jl")
include("./test_skanomalydetector.jl")
@everywhere include("./test_caret_tspredictor.jl")
end
