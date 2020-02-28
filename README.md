
| **Documentation** | **Build Status** | **Help** |
|:---:|:---:|:---:|
| [![][docs-dev-img]][docs-dev-url] [![][docs-stable-img]][docs-stable-url] | [![][travis-img]][travis-url] [![][codecov-img]][codecov-url] | [![][slack-img]][slack-url] [![][gitter-img]][gitter-url] |

### AutoMLPipeline 
is a package to make it trivial to create complex ML pipeline structure using simple expressions which paves the way to manipulate its elements to automatically discover optimal pipelines for machine learning prediction and classification.

#### Load the AutMLPipeline package and submodules
```julia
using Random
using AutoMLPipeline
using AutoMLPipeline.FeatureSelectors
using AutoMLPipeline.EnsembleMethods
using AutoMLPipeline.CrossValidators
using AutoMLPipeline.DecisionTreeLearners
using AutoMLPipeline.Pipelines
using AutoMLPipeline.BaseFilters
using AutoMLPipeline.SKPreprocessors
using AutoMLPipeline.Utils
```

#### Load the different elements in a pipeline
```julia
#### Decomposition
pca = SKPreprocessor("PCA")
fa = SKPreprocessor("FactorAnalysis")
ica = SKPreprocessor("FastICA")
#### Scaler 
rb = SKPreprocessor("RobustScaler")
pt = SKPreprocessor("PowerTransformer")
norm = SKPreprocessor("Normalizer")
mx = SKPreprocessor("MinMaxScaler")
#### categorical preprocessing
ohe = OneHotEncoder()
#### Column selector
disc = CatNumDiscriminator()
catf = CatFeatureSelector()
numf = NumFeatureSelector()
#### Learners
rf = SKLearner("RandomForestClassifier")
gb = SKLearner("GradientBoostingClassifier")
lsvc = SKLearner("LinearSVC")
svc = SKLearner("SVC")
mlp = SKLearner("MLPClassifier")
ada = SKLearner("AdaBoostClassifier");
jrf = RandomForest();
vote = VoteEnsemble();
stack = StackEnsemble();
best = BestLearner();
```

#### Load data
```julia
using CSV
profbdata = CSV.read("profb.csv")
X = profbdata[:,2:end] 
Y = profbdata[:,1] |> Vector;
```

#### Votelearner pipeline
```julia
pvote = @pipeline  (catf |> ohe) + (numf) |> vote
pred = fit_transform!(pvote,X,Y)
sc=score(:accuracy,pred,Y)
println(sc)
### cross-validate
crossvalidate(pvote,X,Y,"accuracy_score",5)
```
#### Print corresponding function call
```julia
@pipelinex (catf |> ohe) + (numf) |> vote
```

#### RandomForest learner
```julia
prf = @pipeline  (numf |> rb |> pca) + (numf |> rb |> ica) + (catf |> ohe) + (numf |> rb |> fa) |> rf
pred = fit_transform!(prf,X,Y)
score(:accuracy,pred,Y) |> println
crossvalidate(prf,X,Y,"accuracy_score",5)
```
#### Linear Support Vector for Classification
```julia
plsvc = @pipeline ((numf |> rb |> pca)+(numf |> rb |> fa)+(numf |> rb |> ica)+(catf |> ohe )) |> lsvc
pred = fit_transform!(plsvc,X,Y)
score(:accuracy,pred,Y) |> println
crossvalidate(plsvc,X,Y,"accuracy_score",5)
```

## Feature Requests and Contributions

We welcome contributions, feature requests, and suggestions. Here is the link to open an [issue][issues-url] for any problems you encounter. If you want to contribute, please follow the guidelines in [contributors page][contrib-url].

## Help usage

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
