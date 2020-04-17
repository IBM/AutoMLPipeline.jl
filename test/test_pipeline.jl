module TestPipeline

using Random
using Test
using AutoMLPipeline
using AutoMLPipeline.Pipelines
using AutoMLPipeline.BaseFilters
using AutoMLPipeline.SKPreprocessors
using AutoMLPipeline.DecisionTreeLearners
using AutoMLPipeline.Utils
using AutoMLPipeline.FeatureSelectors

global const data = getiris()
global const features = data[:,1:4]
global const X=data[:,1:5]
const Y=data[:,5] |> Vector
X[!,5]= X[!,5] .|> string


global const ohe = OneHotEncoder()
global const pca = SKPreprocessor("PCA")
global const ica = SKPreprocessor("FastICA")
global const fa = SKPreprocessor("FactorAnalysis")
global const disc = CatNumDiscriminator()
global const catf = CatFeatureSelector()
global const numf = NumFeatureSelector()
global const rf = RandomForest()
global const ada = Adaboost()
global const pt = PrunedTree()

function test_pipeline()
  # test initialization of types
  ohe = OneHotEncoder()
  linear1 = Pipeline(Dict(:name=>"lp",:machines => [ohe]))
  linear2 = Pipeline(Dict(:name=>"lp",:machines => [ohe]))
  combo1 = ComboPipeline(Dict(:machines=>[ohe,ohe]))
  combo2 = ComboPipeline(Dict(:machines=>[linear1,linear2]))
  linear1 = Pipeline([ohe])
  linear2 = Pipeline([ohe])
  combo1 = ComboPipeline([ohe,ohe])
  combo2 = ComboPipeline([linear1,linear2])
  # test fit/transform workflow
  fit!(combo1,X)
  res1=transform!(combo1,X)
  res2=fit_transform!(combo1,X)
  @test (res1 .== res2) |> Matrix |> sum == 2100
  fit!(combo2,X)
  res3=transform!(combo2,X)
  res4=fit_transform!(combo2,X)
  @test (res3 .== res4) |> Matrix |> sum == 2100
  pcombo1 = @pipeline ohe + ohe
  pres1 = fit_transform!(pcombo1,X)
  @test (pres1 .== res1) |> Matrix |> sum == 2100
end
@testset "Pipelines" begin
  Random.seed!(123)
  test_pipeline()
end

function test_sympipeline()
  # test symbolic pipeline expression 
  pcombo2 = @pipeline (pca |> ica) + ica + pca
  @test fit_transform!(pcombo2,features) |> Matrix |> size |> collect |> sum == 162
  pcombo2 = @pipeline pca |> ica |> fa
  @test fit_transform!(pcombo2,features) |> Matrix |> size |> collect |> sum == 154
  pcombo3 = @pipeline disc |> ((catf + numf) + (numf |> pca) + (numf |> ica) + (catf|>ohe)) |> rf
  (fit_transform!(pcombo3,X,Y)  .== Y) |> sum == 150
  pcombo4 = @pipeline (numf |> pca) + (numf |> ica) |> (ada * rf * pt)
  @test crossvalidate(pcombo4,X,Y,"accuracy_score",10,false).mean >= 0.90
  pcombo5 = @pipeline :((numf |> pca) + (numf |> ica) |> (ada * rf * pt))
  @test crossvalidate(pcombo5,X,Y,"accuracy_score",10,false).mean >= 0.90
  expr = :((numf |> pca) + (numf |> ica) |> (ada * rf * pt))
  processexpr!(expr.args)
  @test crossvalidate(eval(expr),X,Y,"accuracy_score",10,false).mean >= 0.90
  expr = :((numf |> pca) + (numf |> ica) |> (ada * rf * pt))
  pcombo6 = sympipeline(expr) |> eval
  @test crossvalidate(pcombo6,X,Y,"accuracy_score",10,false).mean >= 0.90
end
@testset "Symbolic Pipelines" begin
  Random.seed!(123)
  test_sympipeline()
end

end
