module AutoTS

greet() = print("Hello World!")

include("carettspredictor.jl")
using .CaretTSPredictors
export CaretTSPredictor, carettsdriver

include("automlflowtsprediction.jl")
using .AutoMLFlowTSPredictions
export AutoMLFlowTSPrediction
export mlftsdriver

end # module AutoTS
