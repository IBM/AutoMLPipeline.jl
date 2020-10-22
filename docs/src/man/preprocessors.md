# Preprocessors
```@setup preprocessor
ENV["COLUMNS"]=1000
```
The design of AMLP is to allow easy extensibility of its processing elements.
The choice of Scikitlearn preprocessors in this initial release 
is more for demonstration purposes to get a good 
narrative of how the various parts of AMLP
fits together to solve a particular problem. AMLP has been tested
to run with a mixture of transformers and filters from Julia, Scikitlearn,
and R's caret in the same pipeline without issues as long as the interfaces
are properly implemented for each wrapped functions.
As there are loads of preprocessing techniques available, the user is encouraged
to create their own wrappers of their favorite implementations 
to allow them interoperability with the existing AMLP implementations.

### SKPreprocessor Structure

```
    SKPreprocessor(args=Dict(
       :name => "skprep",
       :preprocessor => "PCA",
       :impl_args => Dict()
      )
    )

Helper Function:
   SKPreprocessor(preprocessor::String,args::Dict=Dict())
```
SKPreprocessor maintains a dictionary of pre-processors
and dynamically load them based on the `:preprocessor`
name passed during its initialization. The 
`:impl_args` is a dictionary of parameters to be passed
as arguments to the Scikitlearn preprocessor. 

!!! note

    Please consult the documentation in Scikitlearn 
    for what arguments to pass relative to the chosen preprocessor.

Let's try PCA with 2 components decomposition and random state initialized at 0.
```@example preprocessor
using AutoMLPipeline

iris = getiris()
X=iris[:,1:4]

pca = SKPreprocessor("PCA",Dict(:n_components=>2,:random_state=>0))
respca = fit_transform!(pca,X)
nothing #hide
```
```@repl preprocessor
first(respca,5)
```

Let's try ICA with 3 components decomposition and whitening:
```@example preprocessor
ica = SKPreprocessor("FastICA",Dict(:n_components=>3,:whiten=>true))
resica = fit_transform!(ica,X)
nothing #hide
```
```@repl preprocessor
first(resica,5)
```

To get a listing of available preprocessors, use the `skpreprocessors()` function:
```@repl preprocessor
skpreprocessors()
```
