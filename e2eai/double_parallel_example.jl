# make sure local environment is activated
using Pkg
Pkg.activate(".")


# Symbolic Pipeline Composition
# For parallel search
using AutoMLPipeline
using Distributed
using DataFrames

# Add workers
nprocs() ==1 && addprocs(exeflags=["--project=$(Base.active_project())"])
workers()

# disable warnings
@everywhere import PythonCall
@everywhere const PYC=PythonCall
@everywhere warnings = PYC.pyimport("warnings")
@everywhere warnings.filterwarnings("ignore")

@sync @everywhere using AutoMLPipeline
@sync @everywhere using DataFrames

# get data
begin
    data = getprofb()
    X    = data[:,2:end]
    Y    = data[:,1] |> Vector
end

#### feature selectors
catf   = CatFeatureSelector();
numf   = NumFeatureSelector();
# hot-bit encoder
ohe    = OneHotEncoder();
#### feature scalers
rb     = SKPreprocessor("RobustScaler");
pt     = SKPreprocessor("PowerTransformer");
mx     = SKPreprocessor("MinMaxScaler");
std    = SKPreprocessor("StandardScaler");
norm   = SKPreprocessor("Normalizer");
#### feature extractors
pca    = SKPreprocessor("PCA", Dict(:autocomponent => true));
ica    = SKPreprocessor("FastICA", Dict(:autocomponent => true));
fa     = SKPreprocessor("FactorAnalysis", Dict(:autocomponent => true));
#### Learners
rf     = SKLearner("RandomForestClassifier", Dict(:impl_args => Dict(:n_estimators => 10)));
gb     = SKLearner("GradientBoostingClassifier");
lsvc   = SKLearner("LinearSVC");
mlp    = SKLearner("MLPClassifier");
stack  = StackEnsemble();
rbfsvc = SKLearner("SVC");
ada    = SKLearner("AdaBoostClassifier");
vote   = VoteEnsemble();
best   = BestLearner();
tree   = PrunedTree();
sgd    = SKLearner("SGDClassifier");
noop = Identity(Dict(:name => "Noop"));

pipe = @pipeline catf;
pred = fit_transform!(pipe, X, Y)

pipe = @pipeline catf |> ohe;
pred = fit_transform!(pipe, X, Y)

pipe = @pipeline numf;
pred = fit_transform!(pipe, X, Y)

pipe = @pipeline numf |> norm;
pred = fit_transform!(pipe, X, Y)

pipe = @pipeline (numf |> norm) + (catf |> ohe);
pred = fit_transform!(pipe, X, Y)

pipe = @pipeline (numf |> norm) + (catf |> ohe) |> rf;
pred = fit_transform!(pipe, X, Y)
crossvalidate(pipe,X,Y)

pipe = @pipeline (numf |> norm) + (catf |> ohe)  |> sgd;
crossvalidate(pipe,X,Y)

pipe = @pipeline (numf |> norm |> pca) + (numf |> rb |> pca) + (catf |> ohe) |> tree;
crossvalidate(pipe,X,Y)


# Parallel Search for Datamining Optimal Pipelines
function prpsearch()
    learners = [rf,ada,sgd,tree,rbfsvc,lsvc,gb];
    scalers = [rb,pt,norm,std,mx,noop];
    extractors = [pca,ica,fa,noop];
    dftable = @sync @distributed (vcat) for lr in learners
	 @distributed (vcat) for sc in scalers
		@distributed (vcat) for xt  in extractors
		  pipe  = @pipeline (catf |> ohe) + (numf |> sc |> xt)  |> lr
		  scn   = sc.name[1:end - 4]; xtn = xt.name[1:end - 4]; lrn = lr.name[1:end - 4]
		  pname = "$scn |> $xtn |> $lrn"
		  ptime = @elapsed begin
			 mean, sd, kfold, _ = crossvalidate(pipe, X, Y, "accuracy_score", 3)
		  end
		  DataFrame(pipeline=pname, mean=mean, sd=sd, time=ptime, folds=kfold)
		end
	 end
    end
    sort!(dftable, :mean, rev=true);
    dftable
end
runtime = @elapsed begin
    df = prpsearch()
end;
serialtime = df.time |> sum;
(serialtime = "$(round(serialtime / 60.0)) minutes", paralleltime = "$(round(runtime)) seconds")

# pipeline performances
@show df
