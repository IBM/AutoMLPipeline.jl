module TestCrossValidator

using Test
using Random
using AutoMLPipeline
using AutoMLPipeline.CrossValidators
using AutoMLPipeline.DecisionTreeLearners
using AutoMLPipeline.Pipelines
using AutoMLPipeline.BaseFilters
using AutoMLPipeline.SKPreprocessors
using AutoMLPipeline.Utils

function test_crossvalidator()
  Random.seed!(123)
  acc(X,Y) = score(:accuracy,X,Y)
  data=getiris()
  X=data[:,1:4] 
  Y=data[:,5] |> Vector{String}
  rf = RandomForest()
  @test crossvalidate(rf,X,Y,acc).mean > 80.0
  Random.seed!(123)
  ppl1 = LinearPipeline([RandomForest()])
  @test crossvalidate(ppl1,X,Y,acc).mean > 80.0
  Random.seed!(123)
  ohe = OneHotEncoder()
  stdsc= SKPreprocessor("StandardScaler")
  ppl2 = LinearPipeline([ohe,stdsc,RandomForest()])
  @test crossvalidate(ppl2,X,Y,acc).mean > 80.0
  Random.seed!(123)
  mpca = SKPreprocessor("PCA")
  mppca = SKPreprocessor("KernelPCA")
  mfa = SKPreprocessor("FactorAnalysis")
  mica = SKPreprocessor("FastICA")
  mrb = SKPreprocessor("RobustScaler")
  ppl3 = LinearPipeline(Dict(:machines=>[mrb,mica,mpca,mppca,RandomForest()]))
  @test crossvalidate(ppl3,X,Y,acc).mean > 80.0
  Random.seed!(123)
  fit!(ppl3,X,Y)
  @test size(transform!(ppl3,X))[1] == length(Y)
  Random.seed!(123)
  ppl5 = LinearPipeline(Dict(:machines=>[mrb,mica,mppca,RandomForest()]))
  @test crossvalidate(ppl5,X,Y,acc).mean > 50.0
end
@testset "CrossValidator" begin
  test_crossvalidator()
end


end
