# Preprocessing
Let us start by loading the `diabetes` dataset:
```@setup preprocessing
using Random
ENV["COLUMNS"]=1000
Random.seed!(123)
```
```@example preprocessing
using AutoMLPipeline
using CSV
diabetesdf = CSV.read(joinpath(dirname(pathof(AutoMLPipeline)),"../data/diabetes.csv"))
X = diabetesdf[:,1:end-1]
Y = diabetesdf[:,end] |> Vector
nothing #hide
```
We can check the data by showing the first 5 rows:
```@repl preprocessing
show5(df)=first(df,5); # show first 5 rows
show5(diabetesdf)
```

This [UCI dataset](https://archive.ics.uci.edu/ml/datasets/diabetes) 
is a collection of diagnostic tests among the Pima Indians 
to investigate whether the patient shows 
sign of diabetes or not based on certain features:
- Number of times pregnant
- Plasma glucose concentration a 2 hours in an oral glucose tolerance test
- Diastolic blood pressure (mm Hg)
- Triceps skin fold thickness (mm)
- 2-Hour serum insulin (mu U/ml)
- Body mass index (weight in kg/(height in m)^2)
- Diabetes pedigree function
- Age (years)
- Class variable (0 or 1) indicating diabetic or not

What is interesting with this dataset is that one or more numeric columns
can be categorical and should be hot-bit encoded. One way to verify is 
to compute the number of unique instances for each column and look for 
columns with relatively smaller count:
```@repl preprocessing
[n=>length(unique(x)) for (n,x) in pairs(eachcol(df))] |> collect
```

Among the input columns, `preg` has only 17 unique instances and it can
be treated as a categorical variable. However, its description indicates
that the feature refers to the number of times the patient is pregnant
and can be considered numerical. With this dilemma, we need to figure
out which representation provides better performance to our classifier.
In order to test the two options, we can use the Feature Discriminator
module to filter and transform the `preg` column to either numeric
or categorical and choose the pipeline with the optimal performance.

### CatNumDiscriminator for Detecting Categorical Numeric Features
*Transform numeric columns with small unique instances to categories.*

Let us use `CatNumDiscriminator` which expects one argument to indicate
the maximum number of unique instances in order to consider a particular
column as categorical. For the sake of this discussion, let us use its 
default value which is 24.
```@example preprocessing
using AutoMLPipeline, AutoMLPipeline.FeatureSelectors
using AutoMLPipeline.EnsembleMethods, AutoMLPipeline.CrossValidators
using AutoMLPipeline.DecisionTreeLearners, AutoMLPipeline.Pipelines
using AutoMLPipeline.BaseFilters, AutoMLPipeline.SKPreprocessors
using AutoMLPipeline.Utils, AutoMLPipeline.SKLearners

disc = CatNumDiscriminator(24)
@pipeline disc
tr_disc = fit_transform!(disc,X,Y)
nothing #hide
```
```@repl preprocessing
show5(tr_disc)
```
You may notice that the `preg` column is converted by the `CatNumDiscriminator`
into `String` type which can be fed to hot-bit encoder to preprocess 
categorical data:
```@example preprocessing
disc = CatNumDiscriminator(24)
catf = CatFeatureSelector()
ohe = OneHotEncoder()
pohe = @pipeline disc |> catf |> ohe
tr_pohe = fit_transform!(pohe,X,Y)
nothing #hide
```
```@repl preprocessing
show5(tr_pohe)
```
We have now converted all categorical data into hot-bit encoded values.

For a typical scenario, one can consider columns with around 3-10 
unique numeric instances to be categorical. 
Using `CatNumDiscriminator`, it is trivial
to convert columns of features with small unique instances into categorical
and hot-bit encode them as shown below. Let us use 5 as the cut-off and any
columns with less than 5 unique instances is converted to hot-bits.
```@repl preprocessing
using DataFrames: DataFrame, nrow,ncol

df = rand(1:3,100,3) |> DataFrame;
show5(df)
disc = CatNumDiscriminator(5);
pohe = @pipeline disc |> catf |> ohe;
tr_pohe = fit_transform!(pohe,df);
show5(tr_pohe)
```

### Concatenating Hot-Bits with PCA of Numeric Columns

Going back to the original `diabetes` dataset, we can now use the 
`CatNumDiscriminator` to differentiate between categorical 
columns and numerical columns and preprocess them based on their 
types (String vs Number). Below is the pipeline to convert `preg`
column to hot-bits and use PCA for the numerical features:
```@example preprocessing
pca = SKPreprocessor("PCA")
disc = CatNumDiscriminator(24)
ohe = OneHotEncoder()
catf = CatFeatureSelector()
numf = NumFeatureSelector()
pl = @pipeline disc |> ((numf |> pca) + (catf |> ohe))
res_pl = fit_transform!(pl,X,Y)
nothing #hide
```
```@repl preprocessing
show5(res_pl)
```

### Performance Evaluation

Let us compare the RF cross-validation result between two options:
- `preg` column should be categorical vs
- `preg` column is numerical
in predicting diabetes where numerical values are scaled by robust scaler and
decomposed by PCA.

##### Option 1: Assume All Numeric Columns as not Categorical and Evaluate
```@example preprocessing
pca = SKPreprocessor("PCA")
dt = SKLearner("DecisionTreeClassifier")
rf = SKLearner("RandomForestClassifier")
rbs = SKPreprocessor("RobustScaler")
jrf = RandomForest()
lsvc = SKLearner("LinearSVC")
ohe = OneHotEncoder()
catf = CatFeatureSelector()
numf = NumFeatureSelector()
disc = CatNumDiscriminator(0) # disable turning numeric to categorical features
pl = @pipeline disc |> ((numf |>  pca) + (catf |> ohe)) |> jrf
nothing #hide
```
```@repl preprocessing
crossvalidate(pl,X,Y,"accuracy_score",30)
```

##### Option 2: Assume as Categorical Numeric Columns <= 24 and Evaluate

```@example preprocessing
disc = CatNumDiscriminator(24) # turning numeric to categorical if unique instances <= 24
pl = @pipeline disc |> ((numf |>  pca) + (catf |> ohe)) |> jrf
nothing #hide
```
```@repl preprocessing
crossvalidate(pl,X,Y,"accuracy_score",30)
```
From this evaluation, `preg` column should be treated as numerical
because the corresponding pipeline got better performance. One
thing to note is the presence of errors in the cross-validation
performance for the pipeline that treats `preg` as categorical
data. The subset of training data during the
kfold validation may contain singularities and evaluation causes
some errors due to hot-bit encoding that increases data sparsity.
The error, however, may be a bug which needs to be addressed in 
the future.
