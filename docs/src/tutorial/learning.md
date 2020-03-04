# Model Training and Learning

Let us continue our discussion by using another dataset. This time, 
let's use CMC dataset that are mostly categorical. 
[CMC](https://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice)
is about asking women of their contraceptive choice. The dataset is composed
of the following features:
```@setup learning
ENV["COLUMNS"]=100
```
```@example learning
using AutoMLPipeline
using CSV
cmcdata = CSV.read(joinpath(dirname(pathof(AutoMLPipeline)),"../data/cmc.csv"));
head(df) = first(df,5)
nothing #hide
```
```@repl learning
head(cmcdata)
```
Let's examine the number of unique instances for each column:
```@repl learning
[n=>length(unique(x)) for (n,x) in eachcol(cmcdata,true)]
```
Except for Wife's age and Number of children, the other columns
have less than five unique instances. Let's create a pipeline
to filter those columns and convert them to hotbits and 
concatenate them with the standardized scale of the numeric columns.
```@example learning
std = SKPreprocessor("StandardScaler")
ohe = OneHotEncoder()
kohe = SKPreprocessor("OneHotEncoder")
catf = CatFeatureSelector()
numf = NumFeatureSelector()
disc = CatNumDiscriminator(5) # unique instances <= 5 are categories

pcmc = @pipeline disc
```
