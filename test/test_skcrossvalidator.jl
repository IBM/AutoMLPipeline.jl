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

function crossv(ppl,X,Y,folds,verbose)
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

function test_skcrossvalidator()
  data=getiris()
  X=data[:,1:4]
  Y=data[:,5] |> Vector{String}
  ppl1 = Pipeline(Dict(:machines=>[RandomForest()]))
  crossv(ppl1,X,Y,10,false)
  ppl2 = Pipeline(Dict(:machines=>[VoteEnsemble()]))
  crossv(ppl2,X,Y,10,false)
  cat = CatFeatureSelector()
  num = NumFeatureSelector()
  pca = SKPreprocessor("PCA")
  ptf = SKPreprocessor("PowerTransformer")
  rbc = SKPreprocessor("RobustScaler")
  ppl3=@pipeline ((cat + num) + (num |> ptf) + (num |> rbc) + (num |> pca)) |> VoteEnsemble()
  crossv(ppl3,X,Y,10,false)
end
@testset "CrossValidator" begin
  Random.seed!(123)
  test_skcrossvalidator()
end

end
