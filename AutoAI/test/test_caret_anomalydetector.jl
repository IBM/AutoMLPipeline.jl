module TestCaretAD
using Test
using Distributed
using AutoAI
using DataFrames: DataFrame
using Serialization

clearners = keys(AutoAI.CaretAnomalyDetectors.caretadlearner_dict) |> collect

function testcaretad()
  df = rand(100, 3) |> x -> DataFrame(x, :auto)
  dfres = @sync @distributed (hcat) for learner in clearners
    model = CaretAnomalyDetector(learner)
    res = fit_transform!(model, df)
    mname = string(learner)
    DataFrame(mname => res; makeunique=true)
  end
  return dfres
end

@testset "anomaly detection" begin
  df = @sync testcaretad()
  @test count(>=(0), Matrix(df)) == 1200
end

end
