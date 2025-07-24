module TestCaretTSPredictor
using Distributed
using Test
using AutoAI
using DataFrames: DataFrame
using Serialization

const clearners = ["arima", "ets", "theta", "rf_cds_dt", "grand_means", "croston"]

function testcarettspredictors()
  df = rand(100, 1) |> x -> DataFrame(x, :auto)
  tab = @sync @distributed (hcat) for learner in clearners
    model = CaretTSPredictor(learner)
    res = fit_transform!(model, df)
    res
  end
  return tab
end

@testset "test each ts predictor" begin
  res = @sync testcarettspredictors()
  @test count(x -> typeof(x) <: Real, res) == 10 * length(clearners)
end

end
