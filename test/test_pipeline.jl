module TestPipeline

using Test
using AutoMLPipeline
using AutoMLPipeline.Pipelines
using AutoMLPipeline.BaseFilters
using AutoMLPipeline.SKPreprocessors
using AutoMLPipeline.DecisionTreeLearners
using AutoMLPipeline.Utils
using AutoMLPipeline.FeatureSelectors
 

function test_pipeline()
  data = getiris()
  X=data[:,1:5]
  Y=data[:,5] |> Vector
  X[!,5]= X[!,5] .|> string
  ohe = OneHotEncoder()
  ohe1 = OneHotEncoder()
  linear1 = LinearPipeline(Dict(:name=>"lp",:machines => [ohe]))
  linear2 = LinearPipeline(Dict(:name=>"lp",:machines => [ohe]))
  combo1 = ComboPipeline(Dict(:machines=>[ohe,ohe]))
  combo2 = ComboPipeline(Dict(:machines=>[linear1,linear2]))
  linear1 = LinearPipeline([ohe])
  linear2 = LinearPipeline([ohe])
  combo1 = ComboPipeline([ohe,ohe])
  combo2 = ComboPipeline([linear1,linear2])
  fit!(combo1,X)
  res1=transform!(combo1,X)
  res2=fit_transform!(combo1,X)
  @test (res1 .== res2) |> Matrix |> sum == 2100
  fit!(combo2,X)
  res3=transform!(combo2,X)
  res4=fit_transform!(combo2,X)
  @test (res3 .== res3) |> Matrix |> sum == 2100
  pcombo1 = @pipeline ohe1 * ohe1
  pres1 = fit_transform!(pcombo1,X)
  @test (pres1 .== res1) |> Matrix |> sum == 2100
  features = data[:,1:4]
  pca = SKPreprocessor("PCA")
  ica = SKPreprocessor("FastICA")
  fa = SKPreprocessor("FactorAnalysis")
  pcombo2 = @pipeline (pca + ica)*ica*pca
  @test fit_transform!(pcombo2,features) |> Matrix |> size |> collect |> sum == 162
  pcombo2 = @pipeline pca+ica+fa
  @test fit_transform!(pcombo2,features) |> Matrix |> size |> collect |> sum == 154
  disc = CatNumDiscriminator()
  catf = CatFeatureSelector()
  numf = NumFeatureSelector()
  rf = RandomForest()
  pcombo3 = @pipeline disc + ((catf*numf) * (numf+pca) * (numf+ica) * (catf+ohe)) + rf
  (fit_transform!(pcombo3,X,Y)  .== Y) |> sum == 150
end
@testset "Pipelines" begin
    test_pipeline()
end


end
