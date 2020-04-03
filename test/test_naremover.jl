module TestNARemover

using Test
using Random
using AutoMLPipeline.NARemovers
using AutoMLPipeline.AbsTypes
using DataFrames

function test_naremover()
  Random.seed!(123)
  df  = DataFrame(a=rand([1:3...,missing],100),b=rand([1:9...,missing],100),c=rand([1:20...,missing],100))
  nara = NARemover(0.25)
  @test fit_transform!(nara,df) |> Matrix |> x->sum(skipmissing(x)) == 1546
  narb = NARemover(0.05)
  @test fit_transform!(narb,df) |> Matrix |> x->sum(skipmissing(x)) == 1086
  narc = NARemover(0.01)
  @test fit_transform!(narc,df)  == DataFrame()
end
@testset "NA Remover" begin
  Random.seed!(123)
  test_naremover()
end

end
