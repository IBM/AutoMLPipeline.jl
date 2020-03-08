# Pipeline
There are two types of Pipelines: LinearPipeline and ComboPipeline.
LinearPipeline (commonly called `Pipeline` from now on) performs 
sequential evaluation of `fit_transform!` operation
to each of its elements passing the output of previous element as 
input to the next element iteratively. On the other hand,
`ComboPipeline` performs dataframe concatenation of the final 
outputs of its elements which can be a mixture of transformers, filters, 
learners, or pipelines. 

LinearPipeline uses `|>` symbolic expression while ComboPipeline uses `+`. 
The expression, `a |> b`, is equivalent to `Pipeline(a,b)` function call while
the expression, `a + b`, is equivalent to `ComboPipeline(a,b)`. The
elements `a` and `b` can be transformers, filters, learners or 
pipeline themselves.

### Pipeline Structure
The linear pipeline accepts the following arguments wrapped in a 
`Dictionary` type argument:
- `:name` -> alias name for the pipeline
- `:machines` -> a Vector learners/transformers/pipelines
- `:machine_args` -> arguments to elements of the pipeline

For ease of usage, the following function calls are supported:
- `Pipeline(args::Dict)` # init function
- `Pipeline(Vector{<:Machine},args::Dict=Dict())` # using vectors of learners/transformers
- `Pipeline(machs...)` # using ntuples of learners/transformers

### ComboPipeline Structure
ComboPipeline or feature union pipeline accepts similar arguments
with the linear pipeline and follows similar convenient helper
functions:
- `ComboPipeline(args::Dict)` # init function
- `ComboPipeline(Vector{<:Machine},args::Dict=Dict())` # using vectors of learners/transformers
- `ComboPipeline(machs...)` # using ntuples of learners/transformers

Note: Please refer to the [Pipeline Tutorial](@ref PipelineUsage) for illustrations of their usage.
