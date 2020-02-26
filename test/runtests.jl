module TestAutoMLPipeline
using Test

include("test_pipeline.jl")
include("test_skpreprocessing.jl")
include("test_decisiontree.jl")
include("test_ensemble.jl")
include("test_crossvalidator.jl")
include("test_skcrossvalidator.jl")
#include("test_mlbase.jl")
#include("test_tsclassifier.jl")
#include("test_valdate.jl")
#include("test_statifier.jl")
#include("test_monotonicer.jl")
#include("test_cliwrapper.jl")
#include("test_outliernicer.jl")
#include("test_plotter.jl")
#include("test_normalizer.jl")
#include("test_schemalizer.jl")


end
