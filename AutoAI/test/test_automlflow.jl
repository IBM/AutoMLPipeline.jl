module TestAutoMLFlow
using Test
using AutoAI
using CondaPkg
using Statistics


function test_mlflowclassification()
  url = "http://localhost:8080"
  df = get_iris()
  X = df[:, 1:end-1]
  Y = df[:, end] |> collect

  mlfclass = AutoMLFlowClassification(Dict(:url => url))
  Yc = fit_transform!(mlfclass, X, Y)
  println("accuracy = ", mean(Y .== Yc))

  newmfclass = AutoMLFlowClassification(Dict(:url => url, :impl_args => Dict(:nfolds => 2)))
  Yc = fit_transform!(newmfclass, X, Y)
  println("accuracy = ", mean(Y .== Yc))

  nclass = AutoMLFlowClassification(Dict(:url => url))
  nclass.model[:automodel](; nfolds=2)
  Yc = fit_transform!(nclass, X, Y)
  println("accuracy = ", mean(Y .== Yc))

  # test prediction using exisiting trained model from artifacts
  run_id = mlfclass.model[:run_id]
  newmfclass = AutoMLFlowClassification(Dict(:run_id => run_id, :url => url))
  newmfclass = AutoMLFlowClassification(Dict(:url => url))
  newmfclass(; run_id=run_id)
  Yn = transform!(newmfclass, X)
  println("accuracy = ", mean(Yn .== Y))

end

test_mlflowclassification()

function test_mlfregression()
  url = "http://localhost:8080"
  df = get_iris()
  X = df[:, [1, 2, 3, 5]]
  Y = df[:, 4] |> collect

  mlfreg = AutoMLFlowRegression(Dict(:url => url))
  Yc = fit_transform!(mlfreg, X, Y)
  println("mse = ", mean((Y - Yc) .^ 2))

  newmfreg = AutoMLFlowRegression(Dict(:url => url, :impl_args => Dict(:nfolds => 2)))
  Yn = fit_transform!(newmfreg, X, Y)
  println("mse = ", mean((Y - Yn) .^ 2))

  newmfreg = AutoMLFlowRegression(Dict(:url => url))
  newmfreg.model[:automodel](; nfolds=5)
  Yn = fit_transform!(newmfreg, X, Y)
  println("mse = ", mean((Y - Yn) .^ 2))

  ## test prediction using exisiting trained model from artifacts
  run_id = mlfreg.model[:run_id]
  #run_id = "d7ea4d0582bb4519a96b36efbe1eda6a"
  newmfreg = AutoMLFlowRegression(Dict(:run_id => run_id, :url => url))
  newmfreg = AutoMLFlowRegression(Dict(:url => url))
  newmfreg(; run_id, url)
  Yn = transform!(newmfreg, X)
  println("mse = ", mean((Y - Yn) .^ 2))
end

test_mlfregression()

end
