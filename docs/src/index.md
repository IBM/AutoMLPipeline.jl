```@meta
Author = "Paulito P. Palmes"
```

## AutoMLPipeline (AMLP)
is a package that makes it trivial to create 
complex ML pipeline structures using simple 
expressions. AMLP leverages on the built-in
macro programming features of Julia
to symbolically process, manipulate 
pipeline expressions, and
automatically discover optimal structures 
for machine learning prediction and classification.

To illustrate, a typical machine learning workflow that extracts
numerical features (numf) for ICA (independent component analysis) and 
PCA (principal component analysis) transformations, respectively,
concatentated with the hot-bit encoding (ohe) of categorical 
features (catf) of a given data for RF modeling can be expressed 
in AMLP as:

```julia
julia> model = @pipeline (catf |> ohe) + (numf |> pca) + (numf |> ica) |> rf
julia> fit!(model,Xtrain,Ytrain)
julia> prediction = transform!(model,Xtest)
julia> score(:accuracy,prediction,Ytest)
julia> crossvalidate(model,X,Y,"accuracy_score")
julia> crossvalidate(model,X,Y,"balanced_accuracy_score")
```


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
express pipeline structure helps in the analysis
and implementation of the optimization routines.

The target of future work will be the 
implementations of different pipeline 
optimization algorithms ranging from 
evolutionary approaches, integer
programming (discrete choices of POP elements), 
tree/graph search, and hyper-parameter search.

### Package Features
- Pipeline API that allows high-level description of processing workflow
- Common API wrappers for ML libs including Scikitlearn, DecisionTree, etc
- Symbolic pipeline parsing for easy expression 
  of complexed pipeline structures
- Easily extensible architecture by overloading just two main interfaces: fit! and transform!
- Meta-ensembles that allows composition of 
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
(v1.0) pkg> add AutoMLPipeline
```

or

```julia
julia> using Pkg
julia> pkg"add AutoMLPipeline"
```

or

```julia
julia> using Pkg
julia> Pkg.add("AutoMLPipeline")
```

Once AutoMLPipeline is installed, you can 
load it by:

```julia
julia> using AutoMLPipeline
```

or 

```julia
julia> import AutoMLPipeline
```
Generally, you will need the different learners/transformers and utils in AutoMLPipeline for
time-series processing. 

```julia
using AutoMLPipeline 
using AutoMLPipeline.FeatureSelectors
using AutoMLPipeline.EnsembleMethods
using AutoMLPipeline.CrossValidators 
using AutoMLPipeline.DecisionTreeLearners
using AutoMLPipeline.Pipelines
using AutoMLPipeline.BaseFilters
using AutoMLPipeline.SKPreprocessors 
using AutoMLPipeline.Utils`
```

## Tutorial Outline
```@contents
Pages = [
  "tutorial/pipeline.md",
  "tutorial/preprocessing.md",
  "tutorial/learning.md",
  "tutorial/crossvalidation.md"
]
Depth = 3
```

## Manual Outline
```@contents
Pages = [
  "man/pipeline.md",
  "man/ensemble.md",
  "man/learners.md",
  "man/preprocessing.md"
]
Depth = 3
```

## ML Library
```@contents
Pages = [
  "lib/typesfunctions.md"
]
```

```@index
```
