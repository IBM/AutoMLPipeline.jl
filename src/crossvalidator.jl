module CrossValidators

using Statistics: mean, std

using MLBase: StratifiedKfold, LOOCV,
              RandomSub, StratifiedRandomSub

# standard included modules
using Random
using AutoMLPipeline
using AutoMLPipeline.Utils
using DataFrames: DataFrame

export crossvalidate, holdoutcrossvalidate

function holdoutcrossvalidate(pl::Machine,X::DataFrame,Y::Vector,metric::Function,ntimes=10)
  @assert size(X)[1] == length(Y)
end

"""
    crossvalidate(pl::Machine,X::DataFrame,Y::Vector,pfunc::Function,kfolds=10) 

Run K-fold crossvalidation where:
- `pfunc` is a performance metric
- `X` and `Y` are input and target 
"""
function crossvalidate(pl::Machine,X::DataFrame,Y::Vector,
		       pfunc::Function,nfolds::Int=10,verbose::Bool=true) 
  ## flatten arrays
  @assert size(X)[1] == length(Y)
  rowsize = size(X)[1]
  folds = StratifiedKfold(Y,nfolds) |> collect
  pacc = Float64[]
  fold = 0
  error = 0
  for trndx in folds
    ppl = deepcopy(pl)
    rlist = collect(1:rowsize) # nasty bug if not collected and placed inside setdiff
    tsndx = setdiff(rlist,trndx)
    trX = X[trndx,:] 
    trY = Y[trndx] |> collect
    tstX = X[tsndx,:]
    tstY = Y[tsndx] |> collect
    res = 0.0
    try 
      res = pipe_accuracy(ppl,pfunc,trX,trY,tstX,tstY)
      push!(pacc,res)
      fold += 1
      if verbose == true
	println("fold: ",fold,", ",res)
      end
    catch e
      #println(e)
      error += 1
    end
  end
  if verbose == true
    println("errors: ",error)
  end
  (mean=mean(pacc),std=std(pacc),folds=nfolds,errors=error)
end

function pipe_accuracy(plearner,perf::Function,trX::DataFrame,
		       trY::Vector,tstX::DataFrame,tstY::Vector)
  learner = deepcopy(plearner)
  fit!(learner,trX,trY)
  pred = transform!(learner,tstX)
  res = perf(pred,tstY)
  return res
end


end
