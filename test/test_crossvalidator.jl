module TestCrossValidator

using Test
using Random
using AutoMLPipeline
using AutoMLPipeline.Utils

function test_crossvalidator()
  racc = 50.0
  Random.seed!(123)
  acc(X,Y) = score(:accuracy,X,Y)
  data=getiris()
  X=data[:,1:4] 
  Y=data[:,5] |> Vector
  rf = RandomForest()
  @test crossvalidate(rf,X,Y,acc,10,false).mean > racc
  Random.seed!(123)
  ppl1 = Pipeline([RandomForest()])
  @test crossvalidate(ppl1,X,Y,acc,10,false).mean > racc
  Random.seed!(123)
  ohe = OneHotEncoder()
  stdsc= SKPreprocessor("StandardScaler")
  ppl2 = @pipeline ohe |> stdsc |> rf
  @test crossvalidate(ppl2,X,Y,acc,10,false).mean > racc
  Random.seed!(123)
  mpca = SKPreprocessor("PCA")
  mppca = SKPreprocessor("KernelPCA")
  mfa = SKPreprocessor("FactorAnalysis")
  mica = SKPreprocessor("FastICA")
  mrb = SKPreprocessor("RobustScaler")
  ppl3 = @pipeline mrb |> mica |> mpca |> mppca |> rf
  @test crossvalidate(ppl3,X,Y,acc,10,false).mean > racc
  Random.seed!(123)
  fit!(ppl3,X,Y)
  @test size(transform!(ppl3,X))[1] == length(Y)
  Random.seed!(123)
  ppl5 = @pipeline mrb |> mica |> mppca |> rf
  @test crossvalidate(ppl5,X,Y,acc,10,false).mean > racc
end
@testset "CrossValidator" begin
  test_crossvalidator()
end


end
