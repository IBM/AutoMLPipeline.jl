module CrossValidators

using Statistics: mean, std

# standard included modules
using DataFrames
using Random
using AutoMLPipeline.AbsTypes
using AutoMLPipeline.Utils

export crossvalidate

function crossvalidate(pl::Machine,X::DataFrame,Y::Vector,
		       pfunc::Function,nfolds=10) 
  ## flatten arrays
  @assert size(X)[1] == length(Y)
  rowsize = size(X)[1]
  folds = kfold(rowsize,nfolds)
  pacc = Float64[]
  fold = 0
  error = 0
  for trx in folds
    ppl = deepcopy(pl)
    input = deepcopy(X)
    output = deepcopy(Y) |> Vector{String}
    elements = collect(1:rowsize) # nasty bug if not collected and placed inside setdiff
    tsx = setdiff(elements,trx)
    trX = input[trx,:] 
    trY = output[trx] |> collect
    tstX = input[tsx,:]
    tstY = output[tsx] |> collect
    res = 0.0
    try 
      res = pipe_accuracy(ppl,pfunc,trX,trY,tstX,tstY)
      push!(pacc,res)
      fold += 1
      println("fold: ",fold,", ",res)
    catch e
      error += 1
    end
  end
  println("errors: ",error)
  (mean=mean(pacc),std=std(pacc),folds=nfolds)
end

function pipe_accuracy(plearner,perf::Function,trX::DataFrame,
		       trY::Vector,tstX::DataFrame,tstY::Vector)
  learner = deepcopy(plearner)
  trainX = deepcopy(trX)
  trainY = deepcopy(trY)
  testX = deepcopy(tstX)
  testY = deepcopy(tstY)
  fit!(learner,trainX,trainY)
  pred = transform!(learner,testX)
  res = perf(pred,testY)
  return res
end


end
