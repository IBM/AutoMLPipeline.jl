module TestSKL

using Random
using Test
using AutoMLPipeline
using Statistics
using DataFrames: DataFrame

const IRIS = getiris()
const X = IRIS[:,1:3] |> DataFrame
const XC = IRIS[:,1:4] |> DataFrame
const YC = IRIS[:,5] |> Vector
const Y = IRIS[:,4] |> Vector


const classifiers = [
    "LinearSVC","QDA","MLPClassifier","BernoulliNB",
    "RandomForestClassifier","LDA",
    "NearestCentroid","SVC","LinearSVC","NuSVC","MLPClassifier",
    "RidgeClassifierCV","SGDClassifier","KNeighborsClassifier",
    "GaussianProcessClassifier","DecisionTreeClassifier",
    "PassiveAggressiveClassifier","RidgeClassifier",
    "ExtraTreesClassifier","GradientBoostingClassifier",
    "BaggingClassifier","AdaBoostClassifier","GaussianNB","MultinomialNB",
    "ComplementNB","BernoulliNB"
 ]

const regressors = [
    "SVR",
    "Ridge",
    "RidgeCV",
    "Lasso",
    "ElasticNet",
    "Lars",
    "LassoLars",
    "OrthogonalMatchingPursuit",
    "BayesianRidge",
    "ARDRegression",
    "SGDRegressor",
    "PassiveAggressiveRegressor",
    "KernelRidge",
    "KNeighborsRegressor",
    "RadiusNeighborsRegressor",
    "GaussianProcessRegressor",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "ExtraTreesRegressor",
    "GradientBoostingRegressor",
    "MLPRegressor",
    "AdaBoostRegressor"
]
    	

function fit_test(learner::String,in::DataFrame,out::Vector)
    _learner=SKLearner(Dict(:learner=>learner))
    fit!(_learner,in,out)
    @test _learner.model != Dict()
    return _learner
end

function fit_transform_reg(model::Learner,in::DataFrame,out::Vector)
    @test sum((transform!(model,in) .- out).^2)/length(out) < 2.0
end

@testset "scikit classifiers" begin
    Random.seed!(123)
    for cl in classifiers
	fit_test(cl,XC,YC)
    end
end

@testset "scikit regressors" begin
    Random.seed!(123)
    for rg in regressors
	model=fit_test(rg,X,Y)
	fit_transform_reg(model,X,Y)
    end
end

end
