module TestEnsembleMethods

using Test
using Random
using AutoMLPipeline.AbsTypes
using AutoMLPipeline.Utils
using AutoMLPipeline.EnsembleMethods
using DataFrames

function generateXY()
    Random.seed!(123)
    iris = getiris()
    indx = Random.shuffle(1:nrow(iris))
    features=iris[indx,1:4] 
    sp = iris[indx,5] |> Vector
    (features,sp)
end

function getprediction(model,features,output)
  res = fit_transform!(model,features,output)
  sum(res .== output)/length(output)*100
end

function test_ensembles()
  tstfeatures,tstoutput = generateXY()
  models = [VoteEnsemble(),StackEnsemble(),BestLearner()]
  for model in models
    @test getprediction(model,tstfeatures,tstoutput) > 90.0
  end
end
@testset "Ensemble learners" begin
  test_ensembles()
end
  
end # module
