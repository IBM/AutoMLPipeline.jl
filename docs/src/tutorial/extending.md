# Extending AutoMLPipeline 
```@setup csvreader
ENV["COLUMNS"]=1000
```

Having a meta-ML package sounds ideal  but not practical 
in terms of maintainability and flexibility. 
The metapackage becomes a central point of failure
and bottleneck. It doesn't subscribe to the KISS philosophy of
Unix which encourages decentralization of implementation. As long
as the input and output behavior of transformers and learners
follow a standard format, they should work without  
dependency or communication. By using a consistent input/output
interfaces, the passing of information
among the elements in the pipeline will not bring any
surprises to the receivers and transmitters of information
down the line.

Because AMPL's symbolic pipeline is based on the idea of Linux
pipeline and filters, there is a deliberate effort to follow
as much as possible the KISS philosophy by just using two
interfaces to be overloaded (`fit!` and `transform!`): 
input features should be a DataFrame type while
the target output should be a Vector type. Transformers `fit!`
function expects only one input argument and ignores the target 
argument. On the other hand, the `fit!` function of any learner 
requires both input and target arguments to carry out the 
supervised learning phase. For the `transform!` function, both
learners and transformers expect one input argument that both
use to apply their learned parameters in transforming the input
into either prediction, decomposition, normalization, scaling, etc.

#### AMLP Abstract Types
The AMLP abstract types are composed of the following:
```
abstract type Machine end
abstract type Workflow    <:  Machine  end 
abstract type Computer    <:  Machine  end 
abstract type Learner     <:  Computer end
abstract type Transformer <:  Computer end
```
At the top of the hierarchy is the `Machine` abstraction that supports
two major interfaces: `fit!` and `transform!`.
The abstract `Machine` has two major types: `Computer` and `Workflow`. 
The `Computer` types perform computations suchs as filters, transformers, and filters while
the `Workflow` controls the flow of information. A `Workflow` can be a
sequential flow of information or a combination of information from two
or more workflow. A `Workflow` that provides sequential flow is called
`Pipeline` (or linear pipeline) while the one that combines information
from different workflows is called `ComboPipeline`.

The `Computer` type has two subtypes: `Learner` and `Transformer`. Their main
difference is in the behavior of their `fit!` function. The `Learner`
type learns its parameters by finding a mapping function between its 
`input` and `output` arguments while the
`Transformer` does not require these mapping function to perform its operation. 
The `Transfomer` learns all its parameters by just processing its `input` features.
Both `Transfomer` and `Learner` has similar behaviour in the `transform!` function. Both
apply their learned parameters to transform their `input` into `output`.

#### Extending AMLP by Adding a CSVReader Transformer
Let's extend AMLP by adding CSV reading support embedded in the pipeline.
Instead of passing the data in the pipeline argument, we create
a csv transformer that passes the data to succeeding elements in the pipeline
from a csv file.

```@example csvreader
module FileReaders

using CSV 
using DataFrames: DataFrame, nrow,ncol

using AutoMLPipeline
using AutoMLPipeline.AbsTypes # abstract types (Learners and Transformers)

import AutoMLPipeline.fit!
import AutoMLPipeline.transform!

export fit!, transform!
export CSVReader

# define a user-defined structure for type dispatch
mutable struct CSVReader <: Transformer
   name::String
   model::Dict
   args::Dict

   function CSVReader(args = Dict(:fname=>""))
      fname = args[:fname]
      fname != "" || error("missing filename.")  
      isfile(fname) || error("file does not exist.")
      new(fname,Dict(),args)
   end
end

CSVReader(fname::String) = CSVReader(Dict(:fname=>fname))

# Define fit! which does error checking. You can also make 
# it do nothing and let the transform! function does the
# the checking and loading. The fit! function is only defined
# here to make sure there is a fit! dispatch for CSVReader
# type which is needed in the pipeline call iteration.
function fit!(csvreader::CSVReader, df::DataFrame=DataFrame(), target::Vector=Vector())
   fname = csvreader.name
   isfile(fname) || error("file does not exist.")
   csvreader.model = csvreader.args
end

# define transform which opens the file and returns a dataframe
function transform!(csvreader::CSVReader, df::DataFrame=DataFrame())
   fname = csvreader.name
   df = CSV.File(fname) |> DataFrame
   df != DataFrame() || error("empty dataframe.")
   return df
end
end
nothing #hide
```
Let's now load the FileReaders module together with the other AutoMLPipeline
modules and create a pipeline that includes the csv reader we just created.

```@example csvreader
using DataFrames: DataFrame, nrow,ncol


using AutoMLPipeline, AutoMLPipeline.FeatureSelectors, AutoMLPipeline.EnsembleMethods
using AutoMLPipeline.CrossValidators, AutoMLPipeline.DecisionTreeLearners, AutoMLPipeline.Pipelines
using AutoMLPipeline.BaseFilters, AutoMLPipeline.SKPreprocessors, AutoMLPipeline.Utils
using AutoMLPipeline.SKLearners

using .FileReaders # load from the Main module

#### Column selector
catf = CatFeatureSelector() 
numf = NumFeatureSelector()
pca = SKPreprocessor("PCA")
ohe = OneHotEncoder()

fname = joinpath(dirname(pathof(AutoMLPipeline)),"../data/profb.csv")
csvrdr = CSVReader(Dict(:fname=>fname))

p1 = @pipeline csvrdr |> (catf + numf)
df1 = fit_transform!(p1) # empty argument because input coming from csvreader
nothing #hide
```
```@repl csvreader
first(df1,5)
```
```@example csvreader
p2 = @pipeline csvrdr |> (numf |> pca) + (catf |> ohe)  
df2 = fit_transform!(p2) # empty argument because input coming from csvreader
nothing #hide
```
```@repl csvreader
first(df2,5)
```
With the CSVReader extension, csv files can now be directly processed or loaded inside the pipeline
and can be used with other existing filters and transformers.
