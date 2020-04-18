module TestAutoMLPipeline
using Test

# suppress warnings
@info "suppressing PyCall warnings"
using PyCall
warnings = pyimport("warnings")
warnings.filterwarnings("ignore")

# test modules
include("test_baseline.jl")
include("test_skpreprocessing.jl")
include("test_decisiontree.jl")
include("test_sklearner.jl")
include("test_ensemble.jl")
include("test_crossvalidator.jl")
include("test_skcrossvalidator.jl")
include("test_featureselector.jl")
include("test_pipeline.jl")
include("test_naremover.jl")

#include("test_valdate.jl")
#include("test_mlbase.jl")
#include("test_tsclassifier.jl")
#include("test_statifier.jl")
#include("test_monotonicer.jl")
#include("test_cliwrapper.jl")
#include("test_outliernicer.jl")
#include("test_plotter.jl")
#include("test_normalizer.jl")
#include("test_schemalizer.jl")


end
