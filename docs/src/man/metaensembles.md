# Ensemble Methods

AMPL supports three types of meta-ensembles, namely: 
StackEnsemble, VoteEnsemble, and BestLearner. They
are considered as meta-ensembles because they can contain
other learners including other ensembles as well as
meta-ensembles. They support complex level of hierarchy
depending on the requirements. The most effective way to
show their flexibility is to provide some real examples.

### StackEnsemble
Stack ensemble uses the idea of stacking to train 
learners into two stages. The first stage trains
bottom-level learners for the mapping 
between the input and output. The default is to use
70% of the data. Once the bottom-level learners finish the training, 
the algorithm proceeds to stage 2 which treats the
trained learners as transformers. The output from 
these transformers is used to train the Meta-Learner
(RandomForest, PrunedTree, or Adaboost) using the
remaining 30% of the data. 

The StackEnsemble accepts the following arguments
wrapped in a `Dictionary` type argument:
- `:name` -> alias name of ensemble
- `:learners` -> a vector of learners
- `:stacker` -> the meta-learner (RandomForest, or Adaboost, or PrunedTree)
- `:stacker_training_portion` -> percentage of data for the meta-learner
- `:keep_original_features` -> boolean (whether the original data is included together with the transformed data by the bottom-level learners)

While the init function of StackEnsemble expects an argument of
Dictionary type, it supports the following convenient function signatures:
- `StackEnsemble(Dict(:learners=>...,:stacker=>...))`
- `StackEnsemble([learner1,learner2,...],Dict(:stacker=>...))`
- `StackEnsemble([learner1,learner2,...],stacker=...)`
- `StackEnsemble([learner1,learner2,...])`

To illustrate, let's create some bottom-level learners from Scikitlearn and Julia:
```@example ensemble
using AutoMLPipeline
using DataFrames

gauss = SKLearner("GaussianProcessClassifier")
svc = SKLearner("LinearSVC")
ridge = SKLearner("RidgeClassifier")
jrf = RandomForest() # julia's rf
rfstacker = RandomForest()
stackens = StackEnsemble([gauss,svc,ridge,jrf],stacker=rfstacker)
nothing #hide
```
Let's load some dataset and create a pipeline with the `stackens`
as the learner at the end of the pipeline.
```@example ensemble
using CSV
using Random
Random.seed!(123);

profbdata = CSV.File(joinpath(dirname(pathof(AutoMLPipeline)),"../data/profb.csv")) |> DataFrame
X = profbdata[:,2:end] 
Y = profbdata[:,1] |> Vector;

ohe = OneHotEncoder()
catf = CatFeatureSelector();
numf = NumFeatureSelector()
rb = SKPreprocessor("RobustScaler"); 
pt = SKPreprocessor("PowerTransformer");
pca = SKPreprocessor("PCA"); 
fa = SKPreprocessor("FactorAnalysis"); 
ica = SKPreprocessor("FastICA")
pplstacks = @pipeline  (numf |> rb |> pca) + (numf |> rb |> ica) + 
                       (catf |> ohe) + (numf |> rb |> fa) |> stackens
nothing #hide
```
```@repl ensemble
crossvalidate(pplstacks,X,Y)
```
It is worth noting that stack ensemble is dealing with mixture of libraries consisting of Julia's
Random Forest and Scikitlearn learners.

### VoteEnsemble

Vote ensemble uses similar idea with the Stack Ensemble 
but instead of stacking, it uses voting to get the final
prediction. The first stage involves the collection of 
bottom-level learners being trained to learn
the mapping between input and output. Once they are trained
in a classification problem, they are treated as transformers 
wherein the final output of the ensemble is based on the 
output with the greatest count. It's equivalent to majority 
voting where each learner has one vote based on its prediction
output class.

The VoteEnsemble accepts the following arguments
wrapped inside a `Dictionary` type of argument:
- `:name` -> alias name of ensemble
- `:learners` -> a vector of learners

While the init function of VoteEnsemble expects a Dictionary
type of argument, it also supports the following convenient
helper functions:
- `VoteEnsemble(Dict(:learners=>...,:name=>...))`
- `VoteEnsemble([learner1,learner2,...],Dict(:name=>...))`
- `VoteEnsemble([learner1,learner2,...],name=...)`
- `VoteEnsemble([learner1,learner2,...])`

Let's use the same pipeline but substitute the stack ensemble
with the vote ensemble:
```@example ensemble
Random.seed!(123);

votingens = VoteEnsemble([gauss,svc,ridge,jrf]);
pplvote = @pipeline  (numf |> rb |> pca) + (numf |> rb |> ica) + 
                     (catf |> ohe) + (numf |> rb |> fa) |> votingens;
nothing #hide
```
```@repl ensemble
crossvalidate(pplvote,X,Y)
```

### [BestLearner](@id bestlearner)

The BestLearner ensemble does not perform any 2-stage mapping. What it does is
to cross-validate each learner performance and use the most optimal learner
as the final model. This ensemble can be used to automatically pick the 
most optimal learner in a group of learners included in each argument
based on certain selection criteria.

The BestLearner accepts the following arguments
wrapped in `Dictionary` type argument:
- `:selection_function` ->  Function
- `:score_type`         -> Real
- `:partition_generator` -> Function
- `:learners`            -> Vector of learners
- `:name`                -> alias name of learner
- `:learner_options_grid` -> for hyper-parameter search


The BestLearner supports the following function signatures
aside from Dictionary type argument:
- `BestLearner(Dict(:learners=>...,:name=>...))`
- `BestLearner([learner1,learner2,...],Dict(:name=>...))`
- `BestLearner([learner1,learner2,...],name=...)`
- `BestLearner([learner1,learner2,...])`

Let's use the same pipeline as above but substitute the vote ensemble
with the BestLearner ensemble:
```@example ensemble
Random.seed!(123);

bestens = BestLearner([gauss,svc,ridge,jrf]);
pplbest = @pipeline  (numf |> rb |> pca) + (numf |> rb |> ica) + 
                     (catf |> ohe) + (numf |> rb |> fa) |> bestens;
nothing #hide
```
```@repl ensemble
crossvalidate(pplbest,X,Y)
``` 
