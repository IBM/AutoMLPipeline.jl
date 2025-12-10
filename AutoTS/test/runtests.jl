
module TestAutoTS
using Distributed

nprocs() == 1 && addprocs()
@everywhere using AutoTS
@everywhere include("./test_caret_tspredictor.jl")
end
