
| **Documentation** | **Build Status** | **Help** |
|:---:|:---:|:---:|
| [![][docs-dev-img]][docs-dev-url] [![][docs-stable-img]][docs-stable-url] | [![][travis-img]][travis-url] [![][codecov-img]][codecov-url] | [![][slack-img]][slack-url] [![][gitter-img]][gitter-url] |

### AutoMLPipeline 
is a package that makes it trivial to create complex ML pipeline structures using simple expressions. Using Julia macro programming features, it becomes trivial to symbolically process and manipulate the pipeline expressions and its elements  to automatically discover optimal structures for machine learning prediction and classification.

#### Load the AutoMLPipeline package and submodules
```julia
using AutoMLPipeline, AutoMLPipeline.FeatureSelectors, AutoMLPipeline.EnsembleMethods
using AutoMLPipeline.CrossValidators, AutoMLPipeline.DecisionTreeLearners, AutoMLPipeline.Pipelines
using AutoMLPipeline.BaseFilters, AutoMLPipeline.SKPreprocessors, AutoMLPipeline.Utils
```

#### Load some of filters, transformers, learners to be used in a pipeline
```julia
#### Decomposition
pca = SKPreprocessor("PCA"); fa = SKPreprocessor("FactorAnalysis"); ica = SKPreprocessor("FastICA")

#### Scaler 
rb = SKPreprocessor("RobustScaler"); pt = SKPreprocessor("PowerTransformer"); 
norm = SKPreprocessor("Normalizer"); mx = SKPreprocessor("MinMaxScaler")

#### categorical preprocessing
ohe = OneHotEncoder()

#### Column selector
catf = CatFeatureSelector(); 
numf = NumFeatureSelector()

#### Learners
rf = SKLearner("RandomForestClassifier"); 
gb = SKLearner("GradientBoostingClassifier")
lsvc = SKLearner("LinearSVC");     svc = SKLearner("SVC")
mlp = SKLearner("MLPClassifier");  ada = SKLearner("AdaBoostClassifier")
jrf = RandomForest();              vote = VoteEnsemble();
stack = StackEnsemble();           best = BestLearner();
```

#### Load data
```julia
using CSV
profbdata = CSV.read(joinpath(dirname(pathof(AutoMLPipeline)),"../data/profb.csv"))
X = profbdata[:,2:end] 
Y = profbdata[:,1] |> Vector;
head(x)=first(x,5)
head(profbdata)
```

#### Filter categories and hot-encode them
```julia
pohe = @pipeline catf |> ohe
tr = fit_transform!(pohe,X,Y)
head(tr)
```

#### Filter numeric features, compute ica and pca features, and combine both features
```julia
pdec = @pipeline (numf |> pca) + (numf |> ica)
tr = fit_transform!(pdec,X,Y)
head(tr)
```

#### A pipeline expression example for classification using the Voting Ensemble learner
```julia
# take all categorical columns and hotbit encode each, 
# concatenate them to the numerical features,
# and feed them to the voting ensemble
pvote = @pipeline  (catf |> ohe) + (numf) |> vote
pred = fit_transform!(pvote,X,Y)
sc=score(:accuracy,pred,Y)
println(sc)
### cross-validate
crossvalidate(pvote,X,Y,"accuracy_score",5)
```
#### Print corresponding function call of the pipeline expression
```julia
@pipelinex (catf |> ohe) + (numf) |> vote
# outputs: :(Pipeline(ComboPipeline(Pipeline(catf, ohe), numf), vote))
```

#### Another pipeline example using the RandomForest learner
```julia
# combine the pca, ica, fa of the numerical columns,
# combine them with the hot-bit encoded categorial features
# and feed all to the random forest classifier
prf = @pipeline  (numf |> rb |> pca) + (numf |> rb |> ica) + (catf |> ohe) + (numf |> rb |> fa) |> rf
pred = fit_transform!(prf,X,Y)
score(:accuracy,pred,Y) |> println
crossvalidate(prf,X,Y,"accuracy_score",5)
```
#### A pipeline for the Linear Support Vector for Classification
```julia
plsvc = @pipeline ((numf |> rb |> pca)+(numf |> rb |> fa)+(numf |> rb |> ica)+(catf |> ohe )) |> lsvc
pred = fit_transform!(plsvc,X,Y)
score(:accuracy,pred,Y) |> println
crossvalidate(plsvc,X,Y,"accuracy_score",5)
```

#### Extending AutoMLPipeline
```
# If you want to add your own filter/transformer/learner, it is trivial. 
# Just take note that filters and transformers expect one input argument 
# while learners expect input and output arguments in the fit! function. 
# transform! function always expect one input argument in all cases. 

# First, import the abstract types and define your own mutable structure 
# as subtype of either Learner or Transformer. Also load the DataFrames package

using DataFrames
import AutoMLPipeline.AbsTypes: fit!, transform!  #for function overloading 

export fit!, transform!, MyFilter

# define your filter structure
mutable struct MyFilter <: Transformer
  variables here....
  function MyFilter()
      ....
  end
end

#define your fit! function. 
# filters and transformer ignore Y argument. 
# learners process both X and Y arguments.
function fit!(fl::MyFilter, X::DataFrame, Y::Vector=Vector())
     ....
end

#define your transform! function
function transform!(fl::MyFilter, X::DataFrame)::DataFrame
     ....
end

# Note that the main data interchange format is a dataframe so transform! 
# output should always be a dataframe as well as the input for fit! and transform!.
# This is necessary so that the pipeline passes the dataframe format consistently to
# its filters/transformers/learners. Once you have this filter, you can use 
# it as part of the pipeline together with the other learners and filters.
```

### Feature Requests and Contributions

We welcome contributions, feature requests, and suggestions. Here is the link to open an [issue][issues-url] for any problems you encounter. If you want to contribute, please follow the guidelines in [contributors page][contrib-url].

### Help usage

Usage questions can be posted in:
- [Julia Community](https://julialang.org/community/) 
- [Gitter AutoMLPipeline Community][gitter-url]
- [Julia Discourse forum][discourse-tag-url]


[contrib-url]: https://github.com/IBM/AutoMLPipeline.jl/blob/master/CONTRIBUTORS.md
[issues-url]: https://github.com/IBM/AutoMLPipeline.jl/issues

[discourse-tag-url]: https://discourse.julialang.org/

[gitter-url]: https://gitter.im/AutoMLPipelineLearning/community
[gitter-img]: https://badges.gitter.im/ppalmes/TSML.jl.svg

[slack-img]: https://img.shields.io/badge/chat-on%20slack-yellow.svg
[slack-url]: https://julialang.slack.com


[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://ibm.github.io/AutoMLPipeline.jl/stable/
[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://ibm.github.io/AutoMLPipeline.jl/latest/

[travis-img]: https://travis-ci.org/IBM/AutoMLPipeline.jl.svg?branch=master
[travis-url]: https://travis-ci.org/IBM/AutoMLPipeline.jl

[codecov-img]: https://codecov.io/gh/IBM/AutoMLPipeline.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/IBM/AutoMLPipeline.jl
