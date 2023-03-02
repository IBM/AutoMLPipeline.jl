# activate local env
using Pkg
Pkg.activate(".")

# load packages/modules
using Base.Threads
using Distributed
using CSV
using DataFrames
using AutoMLPipeline
using Random

nprocs() ==1 && addprocs(exeflags=["--project=$(Base.active_project())"])
workers()

# disable warnings
@everywhere import PythonCall
@everywhere const PYC=PythonCall
@everywhere warnings = PYC.pyimport("warnings")
@everywhere warnings.filterwarnings("ignore")


@everywhere using DataFrames
@everywhere using AutoMLPipeline


# get data
profbdata = getprofb()
X = profbdata[:,2:end];
Y = profbdata[:,1] |> Vector;
topdf(x)=first(x,5)
topdf(profbdata)


#### Scaler 
rb = SKPreprocessor("RobustScaler"); 
pt = SKPreprocessor("PowerTransformer"); 
norm = SKPreprocessor("Normalizer"); 
mx = SKPreprocessor("MinMaxScaler");
std = SKPreprocessor("StandardScaler")
disc = CatNumDiscriminator();

#### categorical preprocessing
ohe = OneHotEncoder();
#### Column selector
catf = CatFeatureSelector(); 
numf = NumFeatureSelector();

# load filters
#### Decomposition
apca = SKPreprocessor("PCA",Dict(:autocomponent=>true)); 
pca = SKPreprocessor("PCA"); 
afa = SKPreprocessor("FactorAnalysis",Dict(:autocomponent=>true)); 
fa = SKPreprocessor("FactorAnalysis"); 
aica = SKPreprocessor("FastICA",Dict(:autocomponent=>true));
ica = SKPreprocessor("FastICA");

#### Learners
rf = SKLearner("RandomForestClassifier",Dict(:impl_args=>Dict(:n_estimators => 10))); 
gb = SKLearner("GradientBoostingClassifier");
lsvc = SKLearner("LinearSVC");     
mlp = SKLearner("MLPClassifier");  
jrf = RandomForest(); 
stack = StackEnsemble(); 
rbfsvc = SKLearner("SVC");
ada = SKLearner("AdaBoostClassifier");
vote = VoteEnsemble();
best = BestLearner();
tree = PrunedTree()
sgd = SKLearner("SGDClassifier");


# filter categories and hotbit encode
pohe = @pipeline catf |> ohe |> fa |> lsvc ;
crossvalidate(pohe,X,Y)

# filter numeric and apply pca and ica (unigrams)
pdec = @pipeline (numf |> pca) + (numf |> ica);
tr = fit_transform!(pdec,X,Y)

# filter numeric, apply rb/pt transforms and ica/pca extractions (bigrams)
ppt = @pipeline (numf |> rb |> ica) + (numf |> pt |> pca);
tr = fit_transform!(ppt,X,Y)

# rf learn baseline
rfp1 = @pipeline ( (catf |> ohe) + (catf |> ohe) + (numf) ) |> rf;
crossvalidate(rfp1, X,Y)

# rf learn: bigrams, 2-blocks
rfp2 = @pipeline ((catf |> ohe) + numf |> rb |> ica) + (numf |> pt |> pca) |> ada;
crossvalidate(rfp2, X,Y)

# lsvc learn: bigrams, 3-blocks
plsvc = @pipeline ((numf |> rb |> pca)+(numf |> rb |> fa)+(numf |> rb |> ica)+(catf |> ohe )) |> lsvc;
pred = fit_transform!(plsvc,X,Y)
crossvalidate(plsvc,X,Y)

learners = [jrf,ada,sgd,tree,lsvc]
learners = @distributed (vcat) for learner in learners
  pcmc = @pipeline disc |> ((catf |> ohe) + (numf)) |> rb |> pca |> learner
  println(learner.name)
  mean,sd,_ = crossvalidate(pcmc,X,Y,"accuracy_score",10)
  DataFrame(name=learner.name,mean=mean,sd=sd)
end

sort!(learners,:mean,rev=true)
@show learners;
