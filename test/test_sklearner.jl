module TestSKL


using Test
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
    for cl in classifiers
	fit_test(cl,XC,YC)
    end
end

@testset "scikit regressors" begin
    for rg in regressors
	model=fit_test(rg,X,Y)
	fit_transform_reg(model,X,Y)
    end
end

end
