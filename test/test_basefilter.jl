module TestBaseFilter

using Random
using Test
using AutoMLPipeline
using AutoMLPipeline.BaseFilters
using DataFrames: nrow

function test_basefilter()
  data = getiris()
  ohe  = OneHotEncoder()
  mptr = Imputer()
  @test fit_transform!(ohe,data) |> Matrix |> sum |> round == 2229.0
  @test fit_transform(ohe,data) |> Matrix |> sum |> round == 2229.0
  @test fit_transform!(mptr,data) |> Matrix |> x->x[:,1:4] |> sum |> round == 2079.0
  @test fit_transform(mptr,data) |> Matrix |> x->x[:,1:4] |> sum |> round == 2079.0
  Random.seed!(1)
  data.mss=rand([missing,(1:100)...],nrow(data))
  @test fit_transform!(mptr,data) |> Matrix |> x->x[:,[(1:4)...,6]]  |> sum |> round == 9054.0
  @test fit_transform(mptr,data) |> Matrix |> x->x[:,[(1:4)...,6]]  |> sum |> round == 9054.0
  wrp = Wrapper(Dict(:transformer => OneHotEncoder()))
  @test fit_transform!(wrp,data) |> Matrix |> sum |> round == 9204.0
  @test fit_transform(wrp,data) |> Matrix |> sum |> round == 9204.0
  Random.seed!(1)
  data.mss=rand([missing,(1:100)...],nrow(data))
  wrp = Wrapper(Dict(:transformer => Imputer()))
  @test fit_transform!(wrp,data) |> Matrix |> x->x[:,[(1:4)...,6]]  |> sum |> round == 9054.0
  @test fit_transform(wrp,data) |> Matrix |> x->x[:,[(1:4)...,6]]  |> sum |> round == 9054.0
end
@testset "BaseFilter" begin
  Random.seed!(123)
  test_basefilter()
end

end
