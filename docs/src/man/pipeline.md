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
