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
  ppl = deepcopy(pl)
  input = X 
  output = Y |> Vector{String}
  rowsize = size(input)[1]
  folds = kfold(rowsize,nfolds)
  pacc = Float64[]
  for (fold,trainndx) in enumerate(folds)
    testndx = setdiff(1:rowsize,trainndx)
    trX = input[trainndx,:] 
    trY = output[trainndx] |> collect
    tstX = input[testndx,:]
    tstY = output[testndx] |> collect
    res = pipe_accuracy(ppl,pfunc,trX,trY,tstX,tstY)
    push!(pacc,res)
    println("fold: ",fold,", ",res)
  end
  (mean=mean(pacc),std=std(pacc),folds=nfolds)
end

function pipe_accuracy(plearner,perf,trainX,trainY,testX,testY)
  learner = deepcopy(plearner)
  fit!(learner,trainX,trainY)
  pred = transform!(learner,testX)
  perf(pred,testY)
end


end
