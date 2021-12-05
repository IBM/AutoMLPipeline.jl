module TestAutoMLPipeline
using Test

# suppress warnings
@info "suppressing PyCall warnings"
using PyCall
warnings = pyimport("warnings")
warnings.filterwarnings("ignore")

# test modules
include("test_basefilter.jl")
include("test_skpreprocessing.jl")
include("test_sklearner.jl")
include("test_skcrossvalidator.jl")

end
