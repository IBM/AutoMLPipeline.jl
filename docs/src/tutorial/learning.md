# Model Training and Learning

Let us continue our discussion by using another dataset. This time, 
let's use CMC dataset that are mostly categorical. 
[CMC](https://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice)
is about asking women of their contraceptive choice. The dataset is composed
of the following features:
```@repl learning
using AutoMLPipeline
using CSV
cmcdata = CSV.read(joinpath(dirname(pathof(AutoMLPipeline)),"../data/cmc.csv"));
head(df) = first(df,5);
head(cmcdata)
```
