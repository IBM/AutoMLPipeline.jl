module TestSKCrossValidator

using Test
using Random
using AutoMLPipeline
using AutoMLPipeline.SKCrossValidators
using AutoMLPipeline.DecisionTreeLearners
using AutoMLPipeline.Pipelines
using AutoMLPipeline.Utils

function test_skcrossvalidator()
  Random.seed!(123)
  data=getiris()
  X=data[:,1:4]
  Y=data[:,5] |> Vector{String}
  ppl1 = LinearPipeline(Dict(:machines=>[RandomForest()]))
  @test crossvalidate(ppl1,X,Y,"accuracy_score").mean > 0.80
  @test crossvalidate(ppl1,X,Y,"balanced_accuracy_score").mean > 0.80
  @test crossvalidate(ppl1,X,Y,"cohen_kappa_score").mean > 0.80
  @test crossvalidate(ppl1,X,Y,"jaccard_score","weighted").mean > 0.80
  @test crossvalidate(ppl1,X,Y,"matthews_corrcoef").mean > 0.80
  @test crossvalidate(ppl1,X,Y,"hamming_loss").mean < 0.1
  @test crossvalidate(ppl1,X,Y,"zero_one_loss").mean < 0.1
  @test crossvalidate(ppl1,X,Y,"f1_score","weighted").mean > 0.80
  @test crossvalidate(ppl1,X,Y,"precision_score","weighted").mean > 0.80
  @test crossvalidate(ppl1,X,Y,"recall_score","weighted").mean > 0.80
end
@testset "CrossValidator" begin
  test_skcrossvalidator()
end

end
