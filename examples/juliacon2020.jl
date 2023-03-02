# Brief Intro
# - Paulito Palmes, PhD
# - IBM Research Scientist
# - IBM Dublin Research Lab
# 
# Acknowledgement
# 
# Problem Overview
#   - given a set of pipeline elements (prp):
# 	   - feature selectors (fsl)  : catf, numf
#       - scalers (fsc)            : norm, minmax, std, rb
#       - feature extractors (fxt) : pca, ica, fa
#       - learners (lr)            : rf, xgb, svm
# 
#   - Find optimal  U{fsl |> fsc |> fxt} |> lr 
#   - Note: x |> f => f(x); x + y => U{x,y}
#   - Assume one-block prp:  fsl |> fsc |> fxt 
#       - Optimize: prp1 + prp2 + ... + prpn |> lr
#       - Optimize: matching between {prps} x {lrs} s.t. 
# 		  chosen prp |> lr is optimal
#   - Complexity: n(prps) x n(lrs) 
#     - exponential/combinatorial complexity
# 
# - Two-stage strategy (avoid simultaneous search prps and lrs)
#   - Pick a surrogate pipeline and search best lr in {lrs}
#   - Use lr to search for best prp in {prps}
#   - Reduce complexity from n(prps) x n(lrs) to n(prps) + n(lrs)
#   - limitations: 
#     - only pipeline structure optimization
#     - no hyper-parameter optimization
#     - for classification tasks
# 
# - Current Toolkits
#   sklearn: Pipeline, FeatureUnion, Hyperparam-Optim
#   caret: No pipeline, Hyperparam-Optim
#   lale: sklearn + AutoML (CASH)
#     - bayesian optimization, tree search, random, stochastic, evolutionary
# 
# AutoMLPipeline: A package that makes it trivial to create and evaluate machine 
# 					 learning pipeline architectures. It can be used as building block
# 					 for developing AutoML algorithms similar to lale but in Julia 
# 					 ecosystem.
# 

# -------------------------------
# Sample Workflow
# -------------------------------

# make sure local environment is activated
using Pkg
Pkg.activate(".")

# Symbolic Pipeline Composition
# For parallel search
using AutoMLPipeline
using Distributed
using DataFrames

# disable truncation of dataframes columns
import Base.show
show(df::AbstractDataFrame) = show(df,truncate=0)
show(io::IO,df::AbstractDataFrame) = show(io,df;truncate=0)

# add workers
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
pred = fit_transform!(pipe, X, Y);
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
df
