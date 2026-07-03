module TestOutlierDetector

using Test
using AutoAD
using DataFrames: DataFrame, nrow
using OutlierDetection

struct MinimalDetectorModel <: OutlierDetection.OD.DetectorModel end
struct MinimalDetector <: OutlierDetection.OD.UnsupervisedDetector end
score(X) = dropdims(sum(X, dims=1), dims=1)
OutlierDetection.OD.fit(::MinimalDetector, X::OutlierDetection.OD.Data; verbosity)::OutlierDetection.OD.Fit = MinimalDetectorModel(), score(X)
OutlierDetection.OD.transform(::MinimalDetector, ::MinimalDetectorModel, X::OutlierDetection.OD.Data)::OutlierDetection.OD.Scores = score(X)
OutlierDetection.OD.@default_frontend(MinimalDetector)

const X = DataFrame(x1=[0.0, 0.1, 0.2, 0.15, 9.0], x2=[0.0, 0.0, 0.1, 0.2, 9.0])

@testset "OutlierDetection native wrapper" begin
  od = OutlierDetector(MinimalDetector())
  scores = fit_transform!(od, X)
  @test length(scores) == nrow(X)
  @test eltype(scores) <: Real
  @test all(isfinite, scores)

  copy = fit(OutlierDetector(MinimalDetector()), X)
  @test copy isa OutlierDetector
  @test copy !== od
  @test length(transform(copy, X)) == nrow(X)

  prefitted = OutlierDetector(MinimalDetector())
  @test_throws ArgumentError transform!(prefitted, X)

  labels = fit_transform!(OutlierDetector(MinimalDetector(), output=:label, threshold=0.8), X)
  @test length(labels) == nrow(X)
  @test Set(labels) ⊆ Set(["normal", "outlier"])

  onecol = DataFrame(x=[0.0, 0.1, 10.0, 0.2])
  @test length(fit_transform!(OutlierDetector(MinimalDetector()), onecol)) == nrow(onecol)
end

end
