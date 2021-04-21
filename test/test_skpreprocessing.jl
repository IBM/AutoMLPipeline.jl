module TestSKPreprocessing

using Random
using Test
using AutoMLPipeline
using Statistics
using DataFrames: DataFrame, nrow

Random.seed!(1)

const IRIS = getiris()
extra = rand(150,3) |> x->DataFrame(x,:auto)
const X = hcat(IRIS[:,1:4],extra) 
const Y = IRIS[:,5] |> Vector

# "KernelCenterer","MissingIndicator","KBinsDiscretizer","OneHotEncoder", 
const preprocessors = [
     "DictionaryLearning", "FactorAnalysis", "FastICA", "IncrementalPCA",
     "KernelPCA", "LatentDirichletAllocation", "MiniBatchDictionaryLearning",
     "MiniBatchSparsePCA", "NMF", "PCA", 
     "TruncatedSVD", 
     "VarianceThreshold",
     "SimpleImputer",  
     "Binarizer", "FunctionTransformer",
     "MultiLabelBinarizer", "MaxAbsScaler", "MinMaxScaler", "Normalizer",
     "OrdinalEncoder", "PolynomialFeatures", "PowerTransformer", 
     "QuantileTransformer", "RobustScaler", "StandardScaler"
 ]

function fit_test(preproc::String,in::DataFrame,out::Vector)
	_preproc=SKPreprocessor(Dict(:preprocessor=>preproc))
	fit!(_preproc,in,out)
	prep = fit(_preproc,in,out)
	@test _preproc.model != Dict()
	@test prep.model != Dict()
	return _preproc
end

function transform_test(preproc::String,in::DataFrame,out::Vector)
	_preproc=SKPreprocessor(Dict(:preprocessor=>preproc))
	res = fit_transform!(_preproc,in)
	res1 = fit_transform(_preproc,in)
	@test size(res)[1] == size(out)[1]
	@test size(res1)[1] == size(out)[1]
end

@testset "scikit preprocessors fit test" begin
   Random.seed!(123)
   for cl in preprocessors
      #println(cl)
      fit_test(cl,X,Y)
   end
end

@testset "scikit preprocessors transform test" begin
   Random.seed!(123)
   for cl in preprocessors
      #println(cl)
      transform_test(cl,X,Y)
   end
end

function skptest()
    features = X
    labels = Y

    pca = SKPreprocessor(Dict(:preprocessor=>"PCA",:impl_args=>Dict(:n_components=>3)))
    @test fit_transform!(pca,features) |> x->size(x,2) == 3

    pca = SKPreprocessor("PCA",Dict(:autocomponent=>true))
    @test fit_transform!(pca,features) |> x->size(x,2) == 3

    pca = SKPreprocessor("PCA",Dict(:impl_args=> Dict(:n_components=>3)))
    @test fit_transform!(pca,features) |> x->size(x,2) == 3

    svd = SKPreprocessor(Dict(:preprocessor=>"TruncatedSVD",:impl_args=>Dict(:n_components=>2)))
    @test fit_transform!(svd,features) |> x->size(x,2) == 2

    ica = SKPreprocessor(Dict(:preprocessor=>"FastICA",:impl_args=>Dict(:n_components=>2)))
    @test fit_transform!(ica,features) |> x->size(x,2) == 2

    stdsc = SKPreprocessor("StandardScaler")
    @test abs(mean(fit_transform!(stdsc,features) |> Matrix)) < 0.00001

    minmax = SKPreprocessor("MinMaxScaler")
    @test mean(fit_transform!(minmax,features) |> Matrix) > 0.30

    vote = VoteEnsemble()
    stack = StackEnsemble()
    best = BestLearner()
    cat = CatFeatureSelector()
    num = NumFeatureSelector()
    disc = CatNumDiscriminator()
    ohe = OneHotEncoder()

    mpipeline = Pipeline(Dict(
            :machines => [stdsc,pca,best]
    ))
    pred = fit_transform!(mpipeline,features,labels)
    @test score(:accuracy,pred,labels) > 50.0

    fpipe = @pipeline ((cat + num) + (num + pca))  |> stack
    @test ((fit_transform!(fpipe,features,labels) .== labels) |> sum ) / nrow(features) > 0.50

    fpipe1 = ((cat + num) + (num + pca))  >> stack
    @test ((fit_transform!(fpipe1,features,labels) .== labels) |> sum ) / nrow(features) > 0.50

end
@testset "scikit preprocessor fit/transform test with real data" begin
    Random.seed!(123)
    skptest()
end

end
