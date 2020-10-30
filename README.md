
| **Documentation** | **Build Status** | **Help** |
|:---:|:---:|:---:|
| [![][docs-dev-img]][docs-dev-url] [![][docs-stable-img]][docs-stable-url] | [![][travis-img]][travis-url] [![][codecov-img]][codecov-url] | [![][slack-img]][slack-url] [![][gitter-img]][gitter-url] |

### Stargazers over time

[![Stargazers over time](https://starchart.cc/IBM/AutoMLPipeline.jl.svg)](https://starchart.cc/IBM/AutoMLPipeline.jl)


### AutoMLPipeline
------------------
**AutoMLPipeline** is a package 
that makes it trivial 
to create complex ML pipeline structures 
using simple expressions. It leverages on 
the built-in macro programming features of 
Julia to symbolically process, manipulate 
pipeline expressions, and makes it easy to 
discover optimal structures for machine 
learning regression and classification.

To illustrate, here is a pipeline expression 
and evaluation of a typical machine learning 
workflow that extracts numerical features (`numf`) 
for `ica` (Independent Component Analysis) 
and `pca` (Principal Component Analysis) 
transformations, respectively, concatenated with 
the hot-bit encoding (`ohe`) of categorical 
features (`catf`) of a given data for `rf` (Random Forest) modeling:

```julia
model = @pipeline (catf |> ohe) + (numf |> pca) + (numf |> ica) |> rf
fit!(model,Xtrain,Ytrain)
prediction = transform!(model,Xtest)
score(:accuracy,prediction,Ytest)
crossvalidate(model,X,Y,"balanced_accuracy_score")
```
Just take note that `+` has higher priority than `|>` so if you
are not sure, enclose the operations inside parentheses.
```julia
### these two expressions are the same
@pipeline a |> b + c; @pipeline a |> (b + c)

### these two expressions are the same
@pipeline a + b |> c; @pipeline (a + b) |> c
```

More examples can be found in the 
[examples](https://github.com/IBM/AutoMLPipeline.jl/tree/master/examples) 
folder including optimizing pipelines by multi-threading or distributed computing.

### Motivations
The typical workflow in machine learning
classification or prediction requires
some or combination of the following
preprocessing steps together with modeling:
- feature extraction (e.g. ica, pca, svd)
- feature transformation (e.g. normalization, scaling, ohe)
- feature selection (anova, correlation)
- modeling (rf, adaboost, xgboost, lm, svm, mlp)

Each step has several choices of functions
to use together with their corresponding
parameters. Optimizing the performance of the
entire pipeline is a combinatorial search
of the proper order and combination of preprocessing
steps, optimization of their corresponding
parameters, together with searching for
the optimal model and its hyper-parameters.

Because of close dependencies among various
steps, we can consider the entire process
to be a pipeline optimization problem (POP).
POP requires simultaneous optimization of pipeline
structure and parameter adaptation of its elements.
As a consequence, having an elegant way to
express pipeline structure can help lessen
the complexity in the management and analysis 
of the wide-array of choices of optimization routines.

The target of future work will be the
implementations of different pipeline
optimization algorithms ranging from
evolutionary approaches, integer
programming (discrete choices of POP elements),
tree/graph search, and hyper-parameter search.

### Package Features
- Symbolic pipeline API for easy expression and high-level description 
  of complex pipeline structures and processing workflow
- Common API wrappers for ML libs including Scikitlearn, DecisionTree, etc
- Easily extensible architecture by overloading just two main interfaces: fit! and transform!
- Meta-ensembles that allow composition of
    ensembles of ensembles (recursively if needed)
    for robust prediction routines
- Categorical and numerical feature selectors for
    specialized preprocessing routines based on types

### Installation

AutoMLPipeline is in the Julia Official package registry.
The latest release can be installed at the Julia
prompt using Julia's package management which is triggered
by pressing `]` at the julia prompt:
```julia
julia> ]
pkg> update
pkg> add AutoMLPipeline
```

### Sample Usage
Below outlines some typical way to preprocess and model any dataset.

##### 1. Load Data, Extract Input (X) and Target (Y) 
```julia
# Make sure that the input feature is a dataframe and the target output is a 1-D vector.
using AutoMLPipeline
profbdata = getprofb()
X = profbdata[:,2:end] 
Y = profbdata[:,1] |> Vector;
head(x)=first(x,5)
head(profbdata)

5×7 DataFrame. Omitted printing of 1 columns
│ Row │ Home.Away │ Favorite_Points │ Underdog_Points │ Pointspread │ Favorite_Name │ Underdog_name │
│     │ String    │ Int64           │ Int64           │ Float64     │ String        │ String        │
├─────┼───────────┼─────────────────┼─────────────────┼─────────────┼───────────────┼───────────────┤
│ 1   │ away      │ 27              │ 24              │ 4.0         │ BUF           │ MIA           │
│ 2   │ at_home   │ 17              │ 14              │ 3.0         │ CHI           │ CIN           │
│ 3   │ away      │ 51              │ 0               │ 2.5         │ CLE           │ PIT           │
│ 4   │ at_home   │ 28              │ 0               │ 5.5         │ NO            │ DAL           │
│ 5   │ at_home   │ 38              │ 7               │ 5.5         │ MIN           │ HOU           │
```

#### 2. Load Filters, Transformers, and Learners 
```julia
using AutoMLPipeline

#### Decomposition
pca = SKPreprocessor("PCA")
fa  = SKPreprocessor("FactorAnalysis")
ica = SKPreprocessor("FastICA")

#### Scaler 
rb   = SKPreprocessor("RobustScaler")
pt   = SKPreprocessor("PowerTransformer")
norm = SKPreprocessor("Normalizer")
mx   = SKPreprocessor("MinMaxScaler")

#### categorical preprocessing
ohe = OneHotEncoder()

#### Column selector
catf = CatFeatureSelector()
numf = NumFeatureSelector()

#### Learners
rf       = SKLearner("RandomForestClassifier")
gb       = SKLearner("GradientBoostingClassifier")
lsvc     = SKLearner("LinearSVC")
svc      = SKLearner("SVC")
mlp      = SKLearner("MLPClassifier")
ada      = SKLearner("AdaBoostClassifier")
jrf      = RandomForest()
vote     = VoteEnsemble()
stack    = StackEnsemble()
best     = BestLearner()
skrf_reg = SKLearner("RandomForestRegressor")
skgb_reg = SKLearner("GradientBoostingRegressor")
```

Note: You can get a listing of available `SKPreprocessors` and `SKLearners` by invoking the following functions, respectively: 
- `skpreprocessors()`
- `sklearners()`

#### 3. Filter categories and hot-encode them
```julia
pohe = @pipeline catf |> ohe
tr = fit_transform!(pohe,X,Y)
head(tr)

5×56 DataFrame. Omitted printing of 47 columns
│ Row │ x1      │ x2      │ x3      │ x4      │ x5      │ x6      │ x7      │ x8      │ x9      │
│     │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │ Float64 │
├─────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│ 1   │ 1.0     │ 0.0     │ 0.0     │ 0.0     │ 0.0     │ 0.0     │ 0.0     │ 0.0     │ 0.0     │
│ 2   │ 0.0     │ 1.0     │ 0.0     │ 0.0     │ 0.0     │ 0.0     │ 0.0     │ 0.0     │ 0.0     │
│ 3   │ 0.0     │ 0.0     │ 1.0     │ 0.0     │ 0.0     │ 0.0     │ 0.0     │ 0.0     │ 0.0     │
│ 4   │ 0.0     │ 0.0     │ 0.0     │ 1.0     │ 0.0     │ 0.0     │ 0.0     │ 0.0     │ 0.0     │
│ 5   │ 0.0     │ 0.0     │ 0.0     │ 0.0     │ 1.0     │ 0.0     │ 0.0     │ 0.0     │ 0.0     │
```

#### 4. Numerical Feature Extraction Example 

##### 4.1 Filter numeric features, compute ica and pca features, and combine both features
```julia
pdec = @pipeline (numf |> pca) + (numf |> ica)
tr = fit_transform!(pdec,X,Y)
head(tr)

5×8 DataFrame
│ Row │ x1       │ x2       │ x3       │ x4       │ x1_1       │ x2_1       │ x3_1       │ x4_1       │
│     │ Float64  │ Float64  │ Float64  │ Float64  │ Float64    │ Float64    │ Float64    │ Float64    │
├─────┼──────────┼──────────┼──────────┼──────────┼────────────┼────────────┼────────────┼────────────┤
│ 1   │ 2.47477  │ 7.87074  │ -1.10495 │ 0.902431 │ 0.0168432  │ 0.00319873 │ -0.0467633 │ 0.026742   │
│ 2   │ -5.47113 │ -3.82946 │ -2.08342 │ 1.00524  │ -0.0327947 │ -0.0217808 │ -0.0451314 │ 0.00702006 │
│ 3   │ 30.4068  │ -10.8073 │ -6.12339 │ 0.883938 │ -0.0734292 │ 0.115776   │ -0.0425357 │ 0.0497831  │
│ 4   │ 8.18372  │ -15.507  │ -1.43203 │ 1.08255  │ -0.0656664 │ 0.0368666  │ -0.0457154 │ -0.0192752 │
│ 5   │ 16.6176  │ -6.68636 │ -1.66597 │ 0.978243 │ -0.0338749 │ 0.0643065  │ -0.0461703 │ 0.00671696 │
```

##### 4.2 Filter numeric features, transform to robust and power transform scaling, perform ica and pca, respectively, and combine both
```julia
ppt = @pipeline (numf |> rb |> ica) + (numf |> pt |> pca)
tr = fit_transform!(ppt,X,Y)
head(tr)

5×8 DataFrame
│ Row │ x1          │ x2          │ x3         │ x4        │ x1_1      │ x2_1     │ x3_1       │ x4_1      │
│     │ Float64     │ Float64     │ Float64    │ Float64   │ Float64   │ Float64  │ Float64    │ Float64   │
├─────┼─────────────┼─────────────┼────────────┼───────────┼───────────┼──────────┼────────────┼───────────┤
│ 1   │ -0.00308891 │ -0.0269009  │ -0.0166298 │ 0.0467559 │ -0.64552  │ 1.40289  │ -0.0284468 │ 0.111773  │
│ 2   │ 0.0217799   │ -0.00699717 │ 0.0329868  │ 0.0449952 │ -0.832404 │ 0.475629 │ -1.14881   │ -0.01702  │
│ 3   │ -0.115577   │ -0.0503802  │ 0.0736173  │ 0.0420466 │ 1.54491   │ 1.65258  │ -1.35967   │ -2.57866  │
│ 4   │ -0.0370057  │ 0.0190459   │ 0.065814   │ 0.0454864 │ 1.32065   │ 0.563565 │ -2.05839   │ -0.74898  │
│ 5   │ -0.0643088  │ -0.00711682 │ 0.0340452  │ 0.0459816 │ 1.1223    │ 1.45555  │ -0.88864   │ -0.776195 │
```

#### 5. A Pipeline for the Voting Ensemble Classification
```julia
# take all categorical columns and hot-bit encode each, 
# concatenate them to the numerical features,
# and feed them to the voting ensemble
using AutoMLPipeline.Utils
pvote = @pipeline  (catf |> ohe) + (numf) |> vote
pred = fit_transform!(pvote,X,Y)
sc=score(:accuracy,pred,Y)
println(sc)
crossvalidate(pvote,X,Y,"accuracy_score")

fold: 1, 0.5373134328358209
fold: 2, 0.7014925373134329
fold: 3, 0.5294117647058824
fold: 4, 0.6716417910447762
fold: 5, 0.6716417910447762
fold: 6, 0.6119402985074627
fold: 7, 0.5074626865671642
fold: 8, 0.6323529411764706
fold: 9, 0.6268656716417911
fold: 10, 0.5671641791044776
errors: 0
(mean = 0.6057287093942055, std = 0.06724940684190235, folds = 10, errors = 0)
```
Note: `crossvalidate()` supports the following sklearn's performance metric
#### classification:
- `accuracy_score`, `balanced_accuracy_score`, `cohen_kappa_score`
- `jaccard_score`, `matthews_corrcoef`, `hamming_loss`, `zero_one_loss`
- `f1_score`, `precision_score`, `recall_score`, 
#### regression:
- `mean_squared_error`, `mean_squared_log_error`
- `mean_absolute_error`, `median_absolute_error`
- `r2_score`, `max_error`, `mean_poisson_deviance` 
- `mean_gamma_deviance`, `mean_tweedie_deviance`, 
- `explained_variance_score`

#### 6. Use `@pipelinex` instead of `@pipeline` to print the corresponding function calls in 6
```julia
julia> @pipelinex (catf |> ohe) + (numf) |> vote
:(Pipeline(ComboPipeline(Pipeline(catf, ohe), numf), vote))

# another way is to use @macroexpand with @pipeline
julia> @macroexpand @pipeline (catf |> ohe) + (numf) |> vote
:(Pipeline(ComboPipeline(Pipeline(catf, ohe), numf), vote))
```

#### 7. A Pipeline for the Random Forest (RF) Classification
```julia
# compute the pca, ica, fa of the numerical columns,
# combine them with the hot-bit encoded categorical features
# and feed all to the random forest classifier
prf = @pipeline  (numf |> rb |> pca) + (numf |> rb |> ica) + (numf |> rb |> fa) + (catf |> ohe) |> rf
pred = fit_transform!(prf,X,Y)
score(:accuracy,pred,Y) |> println
crossvalidate(prf,X,Y,"accuracy_score")

fold: 1, 0.6119402985074627
fold: 2, 0.7611940298507462
fold: 3, 0.6764705882352942
fold: 4, 0.6716417910447762
fold: 5, 0.6716417910447762
fold: 6, 0.6567164179104478
fold: 7, 0.6268656716417911
fold: 8, 0.7058823529411765
fold: 9, 0.6417910447761194
fold: 10, 0.6865671641791045
errors: 0
(mean = 0.6710711150131694, std = 0.04231869797446545, folds = 10, errors = 0)
```
#### 8. A Pipeline for the Linear Support Vector for Classification (LSVC)
```julia
plsvc = @pipeline ((numf |> rb |> pca)+(numf |> rb |> fa)+(numf |> rb |> ica)+(catf |> ohe )) |> lsvc
pred = fit_transform!(plsvc,X,Y)
score(:accuracy,pred,Y) |> println
crossvalidate(plsvc,X,Y,"accuracy_score")

fold: 1, 0.6567164179104478
fold: 2, 0.7164179104477612
fold: 3, 0.8235294117647058
fold: 4, 0.7164179104477612
fold: 5, 0.7313432835820896
fold: 6, 0.6567164179104478
fold: 7, 0.7164179104477612
fold: 8, 0.7352941176470589
fold: 9, 0.746268656716418
fold: 10, 0.6865671641791045
errors: 0
(mean = 0.7185689201053556, std = 0.04820829087095355, folds = 10, errors = 0)
```

#### 9. A Pipeline for Random Forest Regression
```julia
iris = getiris()
Xreg = iris[:,1:3]
Yreg = iris[:,4] |> Vector
pskrfreg = @pipeline (catf |> ohe) + (numf) |> skrf_reg
res=crossvalidate(pskrfreg,Xreg,Yreg,"mean_absolute_error",10)

fold: 1, 0.1827433333333334
fold: 2, 0.18350888888888886
fold: 3, 0.11627222222222248
fold: 4, 0.1254152380952376
fold: 5, 0.16502333333333377
fold: 6, 0.10900222222222226
fold: 7, 0.12561111111111076
fold: 8, 0.14243000000000025
fold: 9, 0.12130555555555576
fold: 10, 0.18811111111111098
errors: 0
(mean = 0.1459423015873016, std = 0.030924217263958102, folds = 10, errors = 0)
```

Note: More examples can be found in the *test* directory of the package. Since
the code is written in Julia, you are highly encouraged to read the source
code and feel free to extend or adapt the package to your problem. Please
feel free to submit PRs to improve the package features. 

#### 10. Performance Comparison of Several Learners
##### 10.1 Sequential Processing
```julia
using Random
using DataFrames

Random.seed!(1)
jrf = RandomForest()
ada = SKLearner("AdaBoostClassifier")
sgd = SKLearner("SGDClassifier")
tree = PrunedTree()
std = SKPreprocessor("StandardScaler")
disc = CatNumDiscriminator()
lsvc = SKLearner("LinearSVC")

learners = DataFrame()
for learner in [jrf,ada,sgd,tree,lsvc]
   pcmc = @pipeline disc |> ((catf |> ohe) + (numf |> std)) |> learner
   println(learner.name)
   mean,sd,_ = crossvalidate(pcmc,X,Y,"accuracy_score",10)
   global learners = vcat(learners,DataFrame(name=learner.name,mean=mean,sd=sd))
end;
@show learners;

learners = 5×3 DataFrame
│ Row │ name                   │ mean     │ sd        │
│     │ String                 │ Float64  │ Float64   │
├─────┼────────────────────────┼──────────┼───────────┤
│ 1   │ rf_M6x                 │ 0.653424 │ 0.0754433 │
│ 2   │ AdaBoostClassifier_KPx │ 0.69504  │ 0.0514792 │
│ 3   │ SGDClassifier_P0n      │ 0.694908 │ 0.0641564 │
│ 4   │ prunetree_zzO          │ 0.621927 │ 0.0578242 │
│ 5   │ LinearSVC_9l7          │ 0.726097 │ 0.0498317 │
```

##### 10.2 Parallel Processing
```julia
using Random
using DataFrames
using Distributed

nprocs() == 1 && addprocs()
@everywhere using DataFrames
@everywhere using AutoMLPipeline

Random.seed!(1)
jrf = RandomForest()
ada = SKLearner("AdaBoostClassifier")
sgd = SKLearner("SGDClassifier")
tree = PrunedTree()
std = SKPreprocessor("StandardScaler")
disc = CatNumDiscriminator()
lsvc = SKLearner("LinearSVC")

learners = @distributed (vcat) for learner in [jrf,ada,sgd,tree,lsvc]
   pcmc = @pipeline disc |> ((catf |> ohe) + (numf |> std)) |> learner
   println(learner.name)
   mean,sd,_ = crossvalidate(pcmc,X,Y,"accuracy_score",10)
   DataFrame(name=learner.name,mean=mean,sd=sd)
end
@show learners;

      From worker 3:    AdaBoostClassifier_KPx
      From worker 4:    SGDClassifier_P0n
      From worker 5:    prunetree_zzO
      From worker 2:    rf_M6x
      From worker 6:    LinearSVC_9l7
      From worker 4:    fold: 1, 0.6716417910447762
      From worker 5:    fold: 1, 0.6567164179104478
      From worker 6:    fold: 1, 0.6865671641791045
      From worker 2:    fold: 1, 0.7164179104477612
      From worker 4:    fold: 2, 0.7164179104477612
      From worker 5:    fold: 2, 0.6119402985074627
      From worker 6:    fold: 2, 0.8059701492537313
      From worker 2:    fold: 2, 0.6716417910447762
      From worker 4:    fold: 3, 0.6764705882352942
      ....
      
learners = 5×3 DataFrame
│ Row │ name                   │ mean     │ sd        │
│     │ String                 │ Float64  │ Float64   │
├─────┼────────────────────────┼──────────┼───────────┤
│ 1   │ rf_M6x                 │ 0.647388 │ 0.0764844 │
│ 2   │ AdaBoostClassifier_KPx │ 0.712862 │ 0.0471003 │
│ 3   │ SGDClassifier_P0n      │ 0.710009 │ 0.05173   │
│ 4   │ prunetree_zzO          │ 0.60428  │ 0.0403121 │
│ 5   │ LinearSVC_9l7          │ 0.726383 │ 0.0467506 │
```

#### 11. Automatic Selection of Best Learner
You can use `*` operation as a selector function which outputs the result of the best learner.
If we use the same pre-processing pipeline in 10, we expect that the average performance of
best learner which is `lsvc` will be around 73.0.
```julia
Random.seed!(1)
pcmc = @pipeline disc |> ((catf |> ohe) + (numf |> std)) |> (jrf * ada * sgd * tree * lsvc)
crossvalidate(pcmc,X,Y,"accuracy_score",10)

fold: 1, 0.7164179104477612
fold: 2, 0.7910447761194029
fold: 3, 0.6911764705882353
fold: 4, 0.7761194029850746
fold: 5, 0.6567164179104478
fold: 6, 0.7014925373134329
fold: 7, 0.6417910447761194
fold: 8, 0.7058823529411765
fold: 9, 0.746268656716418
fold: 10, 0.835820895522388
errors: 0
(mean = 0.7262730465320456, std = 0.060932268798867976, folds = 10, errors = 0)
```

#### 12. Learners as Transformers
It is also possible to use learners in the middle of expression to serve
as transformers and their outputs become inputs to the final learner as illustrated
below.
```julia
expr = @pipeline ( 
             ((numf |> rb)+(catf |> ohe) |> gb) + ((numf |> rb)+(catf |> ohe) |> rf) 
       ) |> ohe |> ada;                
crossvalidate(expr,X,Y,"accuracy_score")

fold: 1, 0.6567164179104478
fold: 2, 0.5522388059701493
fold: 3, 0.7205882352941176
fold: 4, 0.7313432835820896
fold: 5, 0.6567164179104478
fold: 6, 0.6119402985074627
fold: 7, 0.6119402985074627
fold: 8, 0.6470588235294118
fold: 9, 0.6716417910447762
fold: 10, 0.6119402985074627
errors: 0
(mean = 0.6472124670763829, std = 0.053739947087648336, folds = 10, errors = 0)
```
One can even include selector function as part of transformer preprocessing routine:
```julia
pjrf = @pipeline disc |> ((catf |> ohe) + (numf |> std)) |> 
         ((jrf * ada ) + (sgd * tree * lsvc)) |> ohe |> ada
crossvalidate(pjrf,X,Y,"accuracy_score")

fold: 1, 0.7164179104477612
fold: 2, 0.7164179104477612
fold: 3, 0.7941176470588235
fold: 4, 0.7761194029850746
fold: 5, 0.6268656716417911
fold: 6, 0.6716417910447762
fold: 7, 0.7611940298507462
fold: 8, 0.7352941176470589
fold: 9, 0.7761194029850746
fold: 10, 0.6865671641791045
errors: 0
(mean = 0.7260755048287972, std = 0.0532393731318768, folds = 10, errors = 0)
```
Note: The `ohe` is necessary in both examples
because the outputs of the learners and selector function are categorical 
values that need to be hot-bit encoded before feeding to the final `ada` learner.

#### 13. Tree Visualization of the Pipeline Structure
You can visualize the pipeline by using AbstractTrees Julia package. 
```julia
# package installation 
julia> using Pkg
julia> Pkg.update()
julia> Pkg.add("AbstractTrees") 

# load the packages
julia> using AbstractTrees
julia> using AutoMLPipeline

julia> expr = @pipelinex (catf |> ohe) + (numf |> pca) + (numf |> ica) |> rf
:(Pipeline(ComboPipeline(Pipeline(catf, ohe), Pipeline(numf, pca), Pipeline(numf, ica)), rf))

julia> print_tree(stdout, expr)
:(Pipeline(ComboPipeline(Pipeline(catf, ohe), Pipeline(numf, pca), Pipeline(numf, ica)), rf))
├─ :Pipeline
├─ :(ComboPipeline(Pipeline(catf, ohe), Pipeline(numf, pca), Pipeline(numf, ica)))
│  ├─ :ComboPipeline
│  ├─ :(Pipeline(catf, ohe))
│  │  ├─ :Pipeline
│  │  ├─ :catf
│  │  └─ :ohe
│  ├─ :(Pipeline(numf, pca))
│  │  ├─ :Pipeline
│  │  ├─ :numf
│  │  └─ :pca
│  └─ :(Pipeline(numf, ica))
│     ├─ :Pipeline
│     ├─ :numf
│     └─ :ica
└─ :rf
```

### Extending AutoMLPipeline
If you want to add your own filter or transformer or learner, 
take note that filters and transformers process the  
input features but ignores the output argument. On the other hand,
learners process both their input and output arguments during `fit!`
while `transform!` expects one input argument in all cases. 
First step is to import the abstract types and define your own mutable structure 
as subtype of either Learner or Transformer. Next is to import the `fit!` and
`transform!` functions so that you can overload them. Also, you must 
load the DataFrames package because it is the main format for data processing. 
Finally, implement your own `fit` and `transform` and export them.

```julia
using DataFrames
using AutoMLPipeline.AbsTypes

# import functions for overloading
import AutoMLPipeline.AbsTypes: fit!, transform!   

# export the new definitions for dynamic dispatch
export fit!, transform!, MyFilter

# define your filter structure
mutable struct MyFilter <: Transformer
  name::String
  model::Dict
  args::Dict
  function MyFilter(args::Dict())
      ....
  end
end

# define your fit! function. 
function fit!(fl::MyFilter, inputfeatures::DataFrame, target::Vector=Vector())
     ....
end

#define your transform! function
function transform!(fl::MyFilter, inputfeatures::DataFrame)::DataFrame
     ....
end
```

Note that the main format to exchange data is dataframe which requires `transform!`
output to return a dataframe. The features as input for fit! and transform! shall
be in dataframe format too. This is necessary so that 
the pipeline passes the dataframe format consistently to
its corresponding filters/transformers/learners. Once you have 
this transformer, you can use it as part of the pipeline 
together with the other learners and transformers.

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
[slack-url]: https://julialang.slack.com/


[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://ibm.github.io/AutoMLPipeline.jl/stable/
[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://ibm.github.io/AutoMLPipeline.jl/dev/

[travis-img]: https://travis-ci.com/IBM/AutoMLPipeline.jl.svg?branch=master
[travis-url]: https://travis-ci.com/IBM/AutoMLPipeline.jl

[codecov-img]: https://codecov.io/gh/IBM/AutoMLPipeline.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/IBM/AutoMLPipeline.jl
