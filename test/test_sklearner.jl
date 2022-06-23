module TestSKL

using Random
using Test
using AutoMLPipeline
using Statistics
using DataFrames: DataFrame

const IRIS = getiris()
const X = IRIS[:,1:3] |> DataFrame
const XC = IRIS[:,1:4] |> DataFrame
const YC = IRIS[:,5] |> Vector{String}
const Y = IRIS[:,4] |> Vector{Float64}

const classifiers = [
    "LinearSVC","QuadraticDiscriminantAnalysis","MLPClassifier","BernoulliNB",
    "RandomForestClassifier","LinearDiscriminantAnalysis",
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
   lr = fit(_learner,in,out)
   @test _learner.model != Dict()
   @test lr.model != Dict()
   return _learner
end

function fit_transform_reg(model::Learner,in::DataFrame,out::Vector)
   @test sum((transform!(model,in) .- out).^2)/length(out) < 2.0
   @test sum((transform(model,in) .- out).^2)/length(out) < 2.0
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

function pipeline_test()
   pca = SKPreprocessor("PCA")
   catf = CatFeatureSelector()
   numf = NumFeatureSelector()
   rb = SKPreprocessor("RobustScaler")
   ohe=OneHotEncoder()
   regressor = SKLearner("RandomForestRegressor")
   classifier = SKLearner("RandomForestClassifier",n_estimators=100)
   classifier1 = SKLearner("RandomForestClassifier")(n_estimators=200)
   plr   = @pipeline (catf |> ohe) + (numf |> rb |> pca) |> regressor
   plr1  = (catf |> ohe) + (numf |> rb |> pca) |> regressor
   plc   = @pipeline (catf >> ohe) + (numf >> rb >> pca) |> classifier
   plc1  = (catf >> ohe) + (numf >> rb >> pca) |> classifier
   plc2  = (catf >> ohe) + (numf >> rb >> pca) |> classifier1
   @test crossvalidate(plr,X,Y,"mean_absolute_error",3,false).mean < 0.3
   @test crossvalidate(plr1,X,Y,"mean_absolute_error",3,false).mean < 0.3
   @test crossvalidate(plc,XC,YC,"accuracy_score",3,false).mean > 0.8
   @test crossvalidate(plc1,XC,YC,"accuracy_score",3,false).mean > 0.8
end
@testset "scikit pipeline" begin
   pipeline_test()
end

end
