module TestSKCrossValidator

using Test
using Random
using AutoMLPipeline

function crossval_class(ppl, X, Y, nfolds, verbose)
    @test crossvalidate(ppl, X, Y, "accuracy_score", nfolds, verbose).mean > 0.80
    @test crossvalidate(ppl, X, Y, "balanced_accuracy_score", nfolds, verbose).mean > 0.80
    @test crossvalidate(ppl, X, Y, "cohen_kappa_score", nfolds, verbose).mean > 0.80
    @test crossvalidate(ppl, X, Y, "matthews_corrcoef", nfolds, verbose).mean > 0.80
    @test crossvalidate(ppl, X, Y, "hamming_loss", nfolds, verbose).mean < 0.1
    @test crossvalidate(ppl, X, Y, "zero_one_loss", nfolds, verbose).mean < 0.1
    @test crossvalidate(ppl, X, Y, "jaccard_score", "weighted"; nfolds, verbose).mean > 0.80
    @test crossvalidate(ppl, X, Y, "f1_score", "weighted"; nfolds, verbose).mean > 0.80
    @test crossvalidate(ppl, X, Y, "precision_score", "weighted"; nfolds, verbose).mean > 0.80
    @test crossvalidate(ppl, X, Y, "recall_score", "weighted"; nfolds, verbose).mean > 0.80
end

function crossval_reg(ppl, X, Y, folds, verbose)
    @test crossvalidate(ppl, X, Y, "mean_squared_error", folds, verbose).mean < 0.5
    @test crossvalidate(ppl, X, Y, "mean_squared_log_error", folds, verbose).mean < 0.5
    @test crossvalidate(ppl, X, Y, "mean_absolute_error", folds, verbose).mean < 0.5
    @test crossvalidate(ppl, X, Y, "median_absolute_error", folds, verbose).mean < 0.5
    @test crossvalidate(ppl, X, Y, "r2_score", folds, verbose).mean > 0.50
    @test crossvalidate(ppl, X, Y, "max_error", folds, verbose).mean < 0.7
    @test crossvalidate(ppl, X, Y, "mean_poisson_deviance", folds, verbose).mean < 0.7
    @test crossvalidate(ppl, X, Y, "mean_gamma_deviance", folds, verbose).mean < 0.7
    @test crossvalidate(ppl, X, Y, "mean_tweedie_deviance", folds, verbose).mean < 0.7
    @test crossvalidate(ppl, X, Y, "explained_variance_score", folds, verbose).mean > 0.50
end

function test_skcross_reg()
    data = getiris()
    X = data[:, 1:3]
    Y = data[:, 4]
    ppl1 = Pipeline(Dict(:machines => [RandomForest()]))
    ppl1.name = "(data |> rf)"
    crossval_reg(ppl1, X, Y, 10, false)
    ppl2 = Pipeline(Dict(:machines => [VoteEnsemble()]))
    ppl2.name = "(data |> VoteEnsemble)"
    crossval_reg(ppl2, X, Y, 10, false)
    cat = CatFeatureSelector()
    num = NumFeatureSelector()
    pca = SKPreprocessor("PCA")
    ptf = SKPreprocessor("PowerTransformer")
    rbc = SKPreprocessor("RobustScaler")
    ppl3 = @pipeline ((cat + num) + (num |> ptf) + (num |> rbc) + (num |> pca)) |> VoteEnsemble()
    ppl3.name = "((cat + num) + (num |> ptf) + (num |> rbc) + (num |> pca)) |> VoteEnsemble()"
    crossval_reg(ppl3, X, Y, 10, false)
end
@testset "CrossValidator Regression" begin
    Random.seed!(123)
    test_skcross_reg()
end

function test_skcross_class()
    data = getiris()
    X = data[:, 1:4]
    Y = data[:, 5] |> Vector{String}
    ppl1 = Pipeline(Dict(:machines => [RandomForest()]))
    ppl1.name = "(data |> rf)"
    crossval_class(ppl1, X, Y, 10, false)
    ppl2 = Pipeline(Dict(:machines => [VoteEnsemble()]))
    ppl2.name = "(data |> VoteEnsemble)"
    crossval_class(ppl2, X, Y, 10, false)
    cat = CatFeatureSelector()
    num = NumFeatureSelector()
    pca = SKPreprocessor("PCA")
    ptf = SKPreprocessor("PowerTransformer")
    rbc = SKPreprocessor("RobustScaler")
    ppl3 = @pipeline ((cat + num) + (num |> ptf) + (num |> rbc) + (num |> pca)) |> VoteEnsemble()
    ppl3.name = "((cat + num) + (num |> ptf) + (num |> rbc) + (num |> pca)) |> VoteEnsemble()"
    crossval_class(ppl3, X, Y, 10, false)
end
@testset "CrossValidator Classification" begin
    Random.seed!(123)
    test_skcross_class()
end

function test_crossval_options()
    data = getiris()
    X = data[:, 1:4]
    Y = data[:, 5] |> Vector{String}
    acc(x, y) = score(:accuracy, x, y)
    ppl1 = Pipeline(RandomForest())
    ppl1.name = "(data |> rf)"
    @test crossvalidate(ppl1, X, Y, "accuracy_score", 10, false).mean > 0.90
    @test crossvalidate(ppl1, X, Y, "accuracy_score").mean > 0.90
    @test crossvalidate(ppl1, X, Y, "accuracy_score", 10).mean > 0.90
    @test crossvalidate(ppl1, X, Y, "accuracy_score", false).mean > 0.90
    @test crossvalidate(ppl1, X, Y, "accuracy_score", verbose=false).mean > 0.90
    @test crossvalidate(ppl1, X, Y, metric=acc, verbose=false).mean > 0.90
    @test crossvalidate(ppl1, X, Y, metric=acc, nfolds=5).mean > 0.90
    @test crossvalidate(ppl1, X, Y, acc, 5, true).mean > 0.90
end
@testset "CrossValidator Argument Options" begin
    Random.seed!(123)
    test_crossval_options()
end

end
