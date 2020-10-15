# from discourse discussion with zevelev
using Distributed
addprocs()
@everywhere using AutoMLPipeline, DataFrames

#Get models.
sk= AutoMLPipeline.SKLearners.learner_dict |> keys |> collect;
sk= sk |> x-> sort(x,lt=(x,y)->lowercase(x)<lowercase(y));
m_cl= sk[occursin.("Classifier", sk)];
m_cl= m_cl ∪ sk[occursin.("NB", sk)];
m_cl= m_cl ∪ sk[occursin.("SVC", sk)];
m_cl= m_cl ∪ ["LDA", "QDA"];

iris = AutoMLPipeline.Utils.getiris();
X = iris[:,1:4];
Y = iris[:,end] |> Vector;

# find optimal learners
learners = @distributed (vcat) for m in m_cl 
    learner = SKLearner(m)
    pcmc = AutoMLPipeline.@pipeline learner
    println(learner.name)
    mean,sd,folds,err = crossvalidate(pcmc,X,Y,"accuracy_score",5)
    if !isnan(mean)
      DataFrame(name=learner.name,mean=mean,sd=sd,folds=folds,errors=err)
    else
      DataFrame()
    end
end;
sort!(learners,:mean,rev=true)
@show learners;

# optimized C
results=@distributed (vcat) for C in 1:5
  @distributed (vcat) for gamma = 1:5
    svcmodel  = SKLearner("SVC",Dict(:impl_args=>Dict(:kernel=>"rbf",:C=>C,:gamma=>gamma) ))
    mn,sd,fld,err = crossvalidate(svcmodel,X,Y)
    DataFrame(name=svcmodel.name,mean=mn,sd=sd,C=C,gamma=gamma,folds=fld,errors=err)
  end
end
sort!(results,:mean,rev=true)
@show results

# search best learner by crossvalidation and use it for prediction
learners = SKLearner.(["AdaBoostClassifier","BaggingClassifier","SGDClassifier","SVC","LinearSVC"])
blearner = BestLearner(learners)
crossvalidate(blearner,X,Y,"accuracy_score")
fit!(blearner,X,Y)

