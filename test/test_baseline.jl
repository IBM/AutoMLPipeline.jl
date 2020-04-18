module TestBaseline

using Random
using Test
using DataFrames

using AutoMLPipeline
using AutoMLPipeline.Pipelines
using AutoMLPipeline.BaseFilters
using AutoMLPipeline.SKPreprocessors
using AutoMLPipeline.DecisionTreeLearners
using AutoMLPipeline.Utils
using AutoMLPipeline.FeatureSelectors

function test_baseline()
    Random.seed!(123)
    iris=getiris()
    instances=iris[:,1:4]
    labels=iris[:,5] |> collect
    bl = Baseline()
    fit!(bl,instances,labels)
	 @test bl.model[:choice] == "setosa"
    @test sum(transform!(bl,instances) .== repeat(["setosa"],nrow(iris))) == nrow(iris)
    idy = Identity()
    fit!(idy,instances,labels)
	 @test idy.model == Dict()
    @test (transform!(idy,instances) .== instances) |> Matrix |> sum == 150*4
end
@testset "Baseline Tests" begin
  test_baseline()
end

end
