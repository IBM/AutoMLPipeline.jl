module TestBaseline

using Random
using Test
using DataFrames: nrow

using AutoMLPipeline

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
    @test (transform!(idy,instances) .== instances) |> Matrix |> sum == 150*4
end
@testset "Baseline Tests" begin
  test_baseline()
end

end
