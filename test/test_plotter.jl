module TestPlotter

using Test
using Random
using CSV
using AutoMLPipeline
using AutoMLPipeline.Plotters
using AutoMLPipeline.AbsTypes
using AutoMLPipeline.SKLearners
using AutoMLPipeline.SKPreprocessors
using AutoMLPipeline.Utils
using AutoMLPipeline.BaseFilters
using AutoMLPipeline.Pipelines
using AutoMLPipeline.EnsembleMethods
using AutoMLPipeline.FeatureSelectors
using Statistics
using DataFrames
using Makie
using GLMakie
using AbstractPlotting

function test_plotter()
  Random.seed!(123)
  diabetesdf = CSV.read(joinpath(dirname(pathof(AutoMLPipeline)),"../data/diabetes.csv"))
  X = diabetesdf[:,1:end-1]
  Y = diabetesdf[:,end] |> Vector
  numf = NumFeatureSelector()
  plotter=Plotter("line",[1,2]; color="red")
  pipe=@pipeline numf |> plotter
  fit_transform!(pipe,X,Y)
  # @test add a test
end
@testset "Plotter" begin
  Random.seed!(123)
  test_plotter()
end

end
