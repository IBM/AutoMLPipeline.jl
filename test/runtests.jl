module TestAutoMLPipeline
using Test

# @info "suppressing Python warnings"
import PythonCall
const PYC=PythonCall
warnings = PYC.pyimport("warnings")
warnings.filterwarnings("ignore")

include("test_skpreprocessing.jl")
include("test_sklearner.jl")
include("test_skcrossvalidator.jl")

end
