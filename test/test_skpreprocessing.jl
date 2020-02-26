module TestSKPreprocessing

using Test
using AutoMLPipeline.SKPreprocessors
using AutoMLPipeline.Utils
using Statistics
using DataFrames

const IRIS = getiris()
const X = IRIS[:,1:4] |> DataFrame
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
	@test _preproc.model != Dict()
	return _preproc
end

function transform_test(preproc::String,in::DataFrame,out::Vector)
	_preproc=SKPreprocessor(Dict(:preprocessor=>preproc))
	fit!(_preproc,in,out)
	res = transform!(_preproc,in)
	@test size(res)[1] == size(out)[1]
end

@testset "scikit preprocessors fit test" begin
    for cl in preprocessors
	println(cl)
	fit_test(cl,X,Y)
    end
end

@testset "scikit preprocessors transform test" begin
    for cl in preprocessors
	println(cl)
	transform_test(cl,X,Y)
    end
end

function skptest()
    features = X
    labels = Y

    pca = SKPreprocessor(Dict(:preprocessor=>"PCA",:impl_args=>Dict(:n_components=>3)))
    fit!(pca,features)
    @test transform!(pca,features) |> x->size(x,2) == 3

    svd = SKPreprocessor(Dict(:preprocessor=>"TruncatedSVD",:impl_args=>Dict(:n_components=>2)))
    fit!(svd,features)
    @test transform!(svd,features) |> x->size(x,2) == 2

    ica = SKPreprocessor(Dict(:preprocessor=>"FastICA",:impl_args=>Dict(:n_components=>2)))
    fit!(ica,features)
    @test transform!(ica,features) |> x->size(x,2) == 2

    stdsc = SKPreprocessor(Dict(:preprocessor=>"StandardScaler",:impl_args=>Dict()))
    fit!(stdsc,features)
    @test abs(mean(transform!(stdsc,features) |> Matrix)) < 0.00001

    minmax = SKPreprocessor(Dict(:preprocessor=>"MinMaxScaler",:impl_args=>Dict()))
    fit!(minmax,features)
    @test mean(transform!(minmax,features) |> Matrix) ≈ 0.4486931104833648
end
@testset "scikit preprocessor fit/transform test with real data" begin
    skptest()
end


end
