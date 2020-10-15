# note: this example only works on pure julia implementations of 
# preprocessing elements and learners. Scikitlearn and other
# PyCall functions are not thread-safe and cannot be used inside
# the threads loop.

using AutoMLPipeline
using AutoMLPipeline.Utils
using Base.Threads

begin
  profbdata = getprofb()
  X = profbdata[:,2:end]
  Y = profbdata[:,1] |> Vector;
  head(x)=first(x,5)
  head(profbdata)
end

#### Column selector
catf = CatFeatureSelector();
numf = NumFeatureSelector()
ohe = OneHotEncoder()
#### Learners
rf = RandomForest();              
ada = Adaboost()
dt=PrunedTree()

accuracy(X,Y)=score(:accuracy,X,Y)

acc=[]
learners=[rf,ada,dt]
@threads for i in 1:30
  @threads for lr in learners
    println(lr.name)
    pipe=@pipeline ((catf |> ohe) +(numf )) |> lr
    m=crossvalidate(pipe,X,Y,accuracy,10,true)
    push!(acc,(m...,name=lr.name))
    println(m)
  end
end
acc
