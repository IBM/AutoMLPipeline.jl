# Training and Validation

```@setup learning
using Random
ENV["COLUMNS"]=1000
Random.seed!(123)
```
Let us continue our discussion by using another dataset. This time, 
let's use CMC dataset that are mostly categorical. 
[CMC](https://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice)
is about asking women of their contraceptive choice. The dataset is composed
of the following features:
```@example learning
using AutoMLPipeline
using CSV
cmcdata = CSV.File(joinpath(dirname(pathof(AutoMLPipeline)),"../data/cmc.csv")) |> DataFrame;
X = cmcdata[:,1:end-1]
Y = cmcdata[:,end] .|> string
show5(df) = first(df,5)
nothing #hide
```
```@repl learning
show5(cmcdata)
```
Let's examine the number of unique instances for each column:
```@repl learning
[n=>length(unique(x)) for (n,x) in eachcol(cmcdata,true)]
```
Except for Wife's age and Number of children, the other columns
have less than five unique instances. Let's create a pipeline
to filter those columns and convert them to hot-bits and 
concatenate them with the standardized scale of the numeric columns.
```@example learning
std = SKPreprocessor("StandardScaler")
ohe = OneHotEncoder()
kohe = SKPreprocessor("OneHotEncoder")
catf = CatFeatureSelector()
numf = NumFeatureSelector()
disc = CatNumDiscriminator(5) # unique instances <= 5 are categories
pcmc = @pipeline disc |> ((catf |> ohe) + (numf |> std)) 
dfcmc = fit_transform!(pcmc,X)
nothing #hide
```
```@repl learning
show5(dfcmc)
```
### Evaluate Learners with Same Pipeline
You can get a list of sklearners and skpreprocessors by using the following
function calls: 
```@repl learning
sklearners()
skpreprocessors()
```

Let us evaluate 4 learners using the same preprocessing pipeline:
```@example learning
jrf = RandomForest()
ada = SKLearner("AdaBoostClassifier")
sgd = SKLearner("SGDClassifier")
tree = PrunedTree()
nothing #hide
```
```@example learning
using DataFrames: DataFrame, nrow,ncol

learners = DataFrame() 
for learner in [jrf,ada,sgd,tree]
  pcmc = @pipeline disc |> ((catf |> ohe) + (numf |> std)) |> learner
  println(learner.name)
  mean,sd,folds = crossvalidate(pcmc,X,Y,"accuracy_score",5)
  global learners = vcat(learners,DataFrame(name=learner.name,mean=mean,sd=sd,kfold=folds))
end;
nothing #hide
```
```@repl learning
@show learners;
```
For this particular pipeline, Adaboost has the best performance followed
by RandomForest.

Let's extend the pipeline adding Gradient Boost learner and Robust Scaler.
```@example learning
rbs = SKPreprocessor("RobustScaler")
gb = SKLearner("GradientBoostingClassifier")
learners = DataFrame() 
for learner in [jrf,ada,sgd,tree,gb]
  pcmc = @pipeline disc |> ((catf |> ohe) + (numf |> rbs) + (numf |> std)) |> learner
  println(learner.name)
  mean,sd,folds = crossvalidate(pcmc,X,Y,"accuracy_score",5)
  global learners = vcat(learners,DataFrame(name=learner.name,mean=mean,sd=sd,kfold=folds))
end;
nothing #hide
```
```@repl learning
@show learners;
```
This time, Gradient boost has the best performance.
