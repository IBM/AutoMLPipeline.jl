module TestSKCrossValidator

using Test
using Random
using AutoMLPipeline
using AutoMLPipeline.EnsembleMethods
using AutoMLPipeline.SKPreprocessors
using AutoMLPipeline.SKCrossValidators
using AutoMLPipeline.DecisionTreeLearners
using AutoMLPipeline.Pipelines
using AutoMLPipeline.FeatureSelectors
using AutoMLPipeline.Utils

function crossval_class(ppl,X,Y,folds,verbose)
  @test crossvalidate(ppl,X,Y,"accuracy_score",folds,verbose).mean > 0.80
  @test crossvalidate(ppl,X,Y,"balanced_accuracy_score",folds,verbose).mean > 0.80
  @test crossvalidate(ppl,X,Y,"cohen_kappa_score",folds,verbose).mean > 0.80
  @test crossvalidate(ppl,X,Y,"jaccard_score","weighted",folds,verbose).mean > 0.80
  @test crossvalidate(ppl,X,Y,"matthews_corrcoef",folds,verbose).mean > 0.80
  @test crossvalidate(ppl,X,Y,"hamming_loss",folds,verbose).mean < 0.1
  @test crossvalidate(ppl,X,Y,"zero_one_loss",folds,verbose).mean < 0.1
  @test crossvalidate(ppl,X,Y,"f1_score","weighted",folds,verbose).mean > 0.80
  @test crossvalidate(ppl,X,Y,"precision_score","weighted",folds,verbose).mean > 0.80
  @test crossvalidate(ppl,X,Y,"recall_score","weighted",folds,verbose).mean > 0.80
end

function crossval_reg(ppl,X,Y,folds,verbose)
  @test crossvalidate(ppl,X,Y,"mean_squared_error",folds,verbose).mean < 0.5
  @test crossvalidate(ppl,X,Y,"mean_squared_log_error",folds,verbose).mean < 0.5
  @test crossvalidate(ppl,X,Y,"mean_absolute_error",folds,verbose).mean < 0.5
  @test crossvalidate(ppl,X,Y,"median_absolute_error",folds,verbose).mean < 0.5
  @test crossvalidate(ppl,X,Y,"r2_score",folds,verbose).mean > 0.50
  @test crossvalidate(ppl,X,Y,"max_error",folds,verbose).mean < 0.7
  @test crossvalidate(ppl,X,Y,"mean_poisson_deviance",folds,verbose).mean < 0.7
  @test crossvalidate(ppl,X,Y,"mean_gamma_deviance",folds,verbose).mean < 0.7
  @test crossvalidate(ppl,X,Y,"mean_tweedie_deviance",folds,verbose).mean < 0.7
  @test crossvalidate(ppl,X,Y,"explained_variance_score",folds,verbose).mean > 0.50
end

function test_skcross_reg()
  data=getiris()
  X=data[:,1:3]
  Y=data[:,4] 
  ppl1 = Pipeline(Dict(:machines=>[RandomForest()]))
  crossval_reg(ppl1,X,Y,10,false)
  ppl2 = Pipeline(Dict(:machines=>[VoteEnsemble()]))
  crossval_reg(ppl2,X,Y,10,false)
  cat = CatFeatureSelector()
  num = NumFeatureSelector()
  pca = SKPreprocessor("PCA")
  ptf = SKPreprocessor("PowerTransformer")
  rbc = SKPreprocessor("RobustScaler")
  ppl3=@pipeline ((cat + num) + (num |> ptf) + (num |> rbc) + (num |> pca)) |> VoteEnsemble()
  crossval_reg(ppl3,X,Y,10,false)
end
@testset "CrossValidator Regression" begin
  Random.seed!(123)
  test_skcross_reg()
end

function test_skcross_class()
  data=getiris()
  X=data[:,1:4]
  Y=data[:,5] |> Vector{String}
  ppl1 = Pipeline(Dict(:machines=>[RandomForest()]))
  crossval_class(ppl1,X,Y,10,false)
  ppl2 = Pipeline(Dict(:machines=>[VoteEnsemble()]))
  crossval_class(ppl2,X,Y,10,false)
  cat = CatFeatureSelector()
  num = NumFeatureSelector()
  pca = SKPreprocessor("PCA")
  ptf = SKPreprocessor("PowerTransformer")
  rbc = SKPreprocessor("RobustScaler")
  ppl3=@pipeline ((cat + num) + (num |> ptf) + (num |> rbc) + (num |> pca)) |> VoteEnsemble()
  crossval_class(ppl3,X,Y,10,false)
end
@testset "CrossValidator Classification" begin
  Random.seed!(123)
  test_skcross_class()
end

end
