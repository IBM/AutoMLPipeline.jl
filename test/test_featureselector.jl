module TestFeatureSelectors

using Random
using Test
using CSV
using AutoMLPipeline
using AutoMLPipeline.DecisionTreeLearners
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

function iris_test()
    data = getiris()
    X = data[:,1:5]
    catfeat = FeatureSelector([5])
    numfeat = FeatureSelector([1,2,3,4])
    autocat = CatFeatureSelector()
    autonum = NumFeatureSelector()
    @test (fit_transform!(catfeat,X) .== X[:,5]) |> Matrix |> sum == 150
    @test (fit_transform!(numfeat,X) .== X[:,1:4]) |> Matrix |> sum == 600
    @test (fit_transform!(autocat,X) .== X[:,5]) |> Matrix |> sum == 150
    @test (fit_transform!(autonum,X) .== X[:,1:4]) |> Matrix |> sum == 600
    catnumdata = hcat(X,repeat([1,2,3,4,5],30))
    catnum = CatNumDiscriminator()
    res = fit_transform!(catnum,catnumdata)
    @test eltype(catnumdata[:,6]) <: Number
    @test eltype(catnumdata[:,2]) <: Number
    @test eltype(catnumdata[:,4]) <: Number
    @test eltype(res[:,6]) <: String
    @test eltype(res[:,2]) <: String
    @test eltype(res[:,4]) <: String
    catnumdata = hcat(X,repeat([1,2,3,4,5],30))
    catnum = CatNumDiscriminator(0)
    res = fit_transform!(catnum,catnumdata)
    @test eltype(res[:,6]) <: Number
    catnumdata = hcat(X,repeat([1,2,3,4,5],30))
    catnum = CatNumDiscriminator(5)
    ppp=@pipeline catnum |>  ((autocat |> OneHotEncoder()) + (autonum |> SKPreprocessor("PCA")))
    res=fit_transform!(ppp,catnumdata)
    @test ncol(res) == 12
end
@testset "test feature selectors" begin
    Random.seed!(123)
    iris_test()
end

function diabetes_test()
    Random.seed!(123)
    diabetesdf = CSV.read(joinpath(dirname(pathof(AutoMLPipeline)),"../data/diabetes.csv"))
    X = diabetesdf[:,1:end-1]
    Y = diabetesdf[:,end] |> Vector
    pca = SKPreprocessor("PCA")
    ica = SKPreprocessor("FastICA")
    dt = SKLearner("DecisionTreeClassifier")
    rf = SKLearner("RandomForestClassifier")
    rbs = SKPreprocessor("RobustScaler")
    jrf = RandomForest()
    lsvc = SKLearner("LinearSVC")
    ohe = OneHotEncoder()
    catf = CatFeatureSelector()
    numf = NumFeatureSelector()

    disc = CatNumDiscriminator(0)
    pl = @pipeline disc |> ((numf |>  pca) + (catf |> ohe)) |> jrf
    @test crossvalidate(pl,X,Y,"accuracy_score").mean > 0.60

    pl = @pipeline disc |> ((numf |> rbs |>  pca) + (catf |> ohe)) |> lsvc
    @test crossvalidate(pl,X,Y,"accuracy_score").mean > 0.60

    pl = @pipeline disc |> ((numf |> rbs |>  ica) + (catf |> ohe)) |> rf
    @test crossvalidate(pl,X,Y,"accuracy_score").mean > 0.60

    disc = CatNumDiscriminator(20)
    pl = @pipeline disc |> ( (catf |> ohe)) |> jrf
    @test crossvalidate(pl,X,Y,"accuracy_score",20).mean > 0.60

    disc = CatNumDiscriminator(50)
    pl = @pipeline disc |> ((numf |> rbs |>  pca) + (catf |> ohe)) |> lsvc
    @test crossvalidate(pl,X,Y,"accuracy_score",20).mean > 0.60

    disc = CatNumDiscriminator(100)
    pl = @pipeline disc |> ((numf |> rbs |>  ica) + (catf |> ohe)) |> rf
    @test crossvalidate(pl,X,Y,"accuracy_score",20).mean > 0.60
end
@testset "test feature selectors" begin
    Random.seed!(123)
    diabetes_test()
end

end
