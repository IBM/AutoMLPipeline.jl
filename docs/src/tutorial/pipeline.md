# [Pipeline](@id PipelineUsage)
*A tutorial for using the `@pipeline` expression*

### Dataset
Let us start the tutorial by loading the dataset.
```@setup pipeline
using Random
ENV["COLUMNS"]=1000
Random.seed!(123)
```
```@example pipeline
using AutoMLPipeline
using CSV
profbdata = CSV.read(joinpath(dirname(pathof(AutoMLPipeline)),"../data/profb.csv"))
X = profbdata[:,2:end] 
Y = profbdata[:,1] |> Vector
nothing #hide
```
We can check the data by showing the first 5 rows:
```@repl pipeline
show5(df)=first(df,5); # show first 5 rows
show5(profbdata)
```
This dataset is a collection of pro football scores with the
following variables and their descriptions:
- Home/Away = Favored team is at home or away
- Favorite Points = Points scored by the favored team
- Underdog Points = Points scored by the underdog team
- Pointspread = Oddsmaker's points to handicap the favored team
- Favorite Name = Code for favored team's name
- Underdog name = Code for underdog's name
- Year = 89, 90, or 91

!!! note

    For the purpose of this tutorial, we will use the first column,
    Home vs Away, as the target variable to be predicted using the
    other columns as input features. For this target output, we are
    trying to ask whether the model can learn the patterns from its
    input features to predict whether the game was played at home or
    away. Since the input features have both categorical and numerical
    features, the dataset is a good basis to describe 
    how to extract these two types of features, preprocessed them, and
    learn the mapping using a one-liner pipeline expression.

### AutoMLPipeline Modules and Instances
Before continuing further with the tutorial, let us load the 
necessary modules of AutoMLPipeline:
```@example pipeline
using AutoMLPipeline, AutoMLPipeline.FeatureSelectors
using AutoMLPipeline.EnsembleMethods, AutoMLPipeline.CrossValidators
using AutoMLPipeline.DecisionTreeLearners, AutoMLPipeline.Pipelines
using AutoMLPipeline.BaseFilters, AutoMLPipeline.SKPreprocessors
using AutoMLPipeline.Utils, AutoMLPipeline.SKLearners
nothing #hide
```

Let us also create some instances of filters, transformers, and
models that we can use to preprocess and model the dataset.
```@example pipeline
#### Decomposition
pca = SKPreprocessor("PCA"); fa = SKPreprocessor("FactorAnalysis"); 
ica = SKPreprocessor("FastICA")

#### Scaler 
rb = SKPreprocessor("RobustScaler"); pt = SKPreprocessor("PowerTransformer") 
norm = SKPreprocessor("Normalizer"); mx = SKPreprocessor("MinMaxScaler")

#### categorical preprocessing
ohe = OneHotEncoder()

#### Column selector
disc = CatNumDiscriminator()
catf = CatFeatureSelector(); numf = NumFeatureSelector()

#### Learners
rf = SKLearner("RandomForestClassifier"); gb = SKLearner("GradientBoostingClassifier")
lsvc = SKLearner("LinearSVC"); svc = SKLearner("SVC")
mlp = SKLearner("MLPClassifier"); ada = SKLearner("AdaBoostClassifier")
jrf = RandomForest(); vote = VoteEnsemble(); stack = StackEnsemble()           
best = BestLearner()
nothing #hide
```

### Processing Categorical Features
For the first illustration, let us extract categorical features of 
the data and output some of them using the pipeline expression 
and its interface:

```@example pipeline
pop_cat = @pipeline catf 
tr_cat = fit_transform!(pop_cat,X,Y)
nothing #hide
```
```@repl pipeline
show5(tr_cat)
```

One may notice that instead of using `fit!` and `transform`, 
the example uses `fit_transform!` instead. The latter is equivalent
to calling `fit!` and `transform` in sequence which is handy
for examining the final output of the transformation prior to 
feeding it to the model.

Let us now transform the categorical features into one-hot-bit-encoding (ohe)
and examine the results:
```@example pipeline
pop_ohe = @pipeline catf |> ohe
tr_ohe = fit_transform!(pop_ohe,X,Y)
nothing #hide
```
```@repl pipeline
show5(tr_ohe)
```

### Processing Numerical Features
Let us have an example of extracting the numerical features
of the data using different combinations of filters/transformers:
```@example pipeline
pop_rb = @pipeline (numf |> rb)
tr_rb = fit_transform!(pop_rb,X,Y)
nothing #hide
```
```@repl pipeline
show5(tr_rb)
```

### Concatenating Extracted Categorical and Numerical Features
For typical modeling workflow, input features are combinations
of categorical features transformer to one-bit encoding together
with numerical features normalized or scaled or transformed by
decomposition. 

Here is an example of a typical input feature:
```@example pipeline
pop_com = @pipeline (numf |> norm) + (catf |> ohe)
tr_com = fit_transform!(pop_com,X,Y)
nothing #hide
```
```@repl pipeline
show5(tr_com)
```

The column size from 6 grew to 60 after the hot-bit encoding was applied
because of the large number of unique instances for the categorical columns. 

### Performance Evaluation of the Pipeline
We can add a model at the end of the pipeline and evaluate
the performance of the entire pipeline by cross-validation.

Let us use a linear SVC model and evaluate using 5-fold cross-validation.
```@repl pipeline
Random.seed!(12345);
pop_lsvc = @pipeline ( (numf |> rb) + (catf |> ohe) + (numf |> pt)) |> lsvc;
tr_lsvc = crossvalidate(pop_lsvc,X,Y,"balanced_accuracy_score",5)
```

What about using Gradient Boosting model?
```@repl pipeline
Random.seed!(12345);
pop_gb = @pipeline ( (numf |> rb) + (catf |> ohe) + (numf |> pt)) |> gb;
tr_gb = crossvalidate(pop_gb,X,Y,"balanced_accuracy_score",5)
```

What about using Random Forest model?
```@repl pipeline
Random.seed!(12345);
pop_rf = @pipeline ( (numf |> rb) + (catf |> ohe) + (numf |> pt)) |> jrf;
tr_rf = crossvalidate(pop_rf,X,Y,"balanced_accuracy_score",5)
```

Let's evaluate several learners which is a typical workflow
in searching for the optimal model.
```@example pipeline
using Random
using DataFrames
using AutoMLPipeline

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
nothing #hide
```
```@repl pipeline
@show learners;
```

!!! note

    It can be inferred from the results that linear SVC has the best performance
    with respect to the different pipelines evaluated.
    The compact expression supported by the 
    pipeline makes testing of the different combination of features 
    and models trivial. It makes performance evaluation  
    of the pipeline easily manageable in a systematic way.

### Learners as Filters
It is also possible to use learners in the middle of 
expression to serve as filters and their outputs become 
input to the final learner as illustrated below.
```@repl pipeline
expr = @pipeline ( 
                   ((numf |> pca) |> gb) + ((numf |> pca) |> jrf) 
                 ) |> (catf |> ohe) |> ada;
                 
crossvalidate(expr,X,Y,"accuracy_score",5)
```
It is important to take note that the expression `(catf |> ohe)`
is necessary because the outputs of the two learners (`gb` and `jrf`) 
are categorical values that need to be hot-bit encoded before 
feeding them to the final `ada` learner.
