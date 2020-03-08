# Learners
```@setup learner
ENV["COLUMNS"]=1000
using PyCall
warnings = pyimport("warnings")
warnings.filterwarnings("ignore")
```
Similar to `SKPreprocessor`, most of the `Learners` in AMLP
for its initial release are based on Scikitlearn libraries.

!!! note


    For more information and specific details of arguments to pass
    and learner's behaviour, please consult the Scikitlearn 
    documentation.

### SKLearner Structure
```
    SKLearner(Dict(
       :name => "sklearner",
       :output => :class,
       :learner => "LinearSVC",
       :impl_args => Dict()
      )
    )

Helper Function:
    SKLearner(learner::String,args::Dict=Dict())
```
SKLearner maintains a dictionary of learners which can
be listed by invoking the function: `sklearners()`
The `:impl_args` is a dictionary of paramters to be
passed as arguments to the Scikitlearn learner.

Let's try loading some learners with some arguments based on Scikitlearn
documentation:

```@repl learner
using AutoMLPipeline
using AutoMLPipeline.Utils

iris = getiris();
X = iris[:,1:4];
Y = iris[:,end] |> Vector;

rf = SKLearner("RandomForestClassifier",Dict(:n_estimators=>30,:random_state=>0));
crossvalidate(rf,X,Y,"accuracy_score",3)

ada = SKLearner("AdaBoostClassifier",Dict(:n_estimators=>20,:random_state=>0));
crossvalidate(ada,X,Y,"accuracy_score",3)

svc = SKLearner("SVC",Dict(:kernel=>"rbf",:random_state=>0,:gamma=>"auto"));
crossvalidate(svc,X,Y,"accuracy_score",3)
```
