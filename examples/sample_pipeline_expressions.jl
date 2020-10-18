# load packages/modules
using DataFrames
using AutoMLPipeline

# get data
profbdata = getprofb()
X = profbdata[:,2:end];
Y = profbdata[:,1] |> Vector;
topdf(x)=first(x,5)
topdf(profbdata)

# load filters
#### Decomposition
pca = SKPreprocessor("PCA"); 
fa = SKPreprocessor("FactorAnalysis"); 
ica = SKPreprocessor("FastICA");

#### Scaler 
rb = SKPreprocessor("RobustScaler"); 
pt = SKPreprocessor("PowerTransformer"); 
norm = SKPreprocessor("Normalizer"); 
mx = SKPreprocessor("MinMaxScaler");

#### categorical preprocessing
ohe = OneHotEncoder()

#### Column selector
catf = CatFeatureSelector(); 
numf = NumFeatureSelector();

#### Learners
rf = SKLearner("RandomForestClassifier",Dict(:impl_args=>Dict(:n_estimators => 100))); 
gb = SKLearner("GradientBoostingClassifier");
lsvc = SKLearner("LinearSVC");     
mlp = SKLearner("MLPClassifier");  
jrf = RandomForest(); 
stack = StackEnsemble(); 
svc = SKLearner("SVC");
ada = SKLearner("AdaBoostClassifier");
vote = VoteEnsemble();
best = BestLearner();


# filter categories and hotbit encode
pohe = @pipeline catf |> ohe;
tr = fit_transform!(pohe,X,Y)


pohe = @pipeline numf |> pca;
tr = fit_transform!(pohe,X,Y)

# filter numeric and apply pca and ica 
pdec = @pipeline (numf |> pca) + (numf |> ica);
tr = fit_transform!(pdec,X,Y)

# filter numeric, apply rb/pt transforms and ica/pca extractions 
ppt = @pipeline (numf |> rb |> ica) + (numf |> pt |> pca) 
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

@pipelinex ((numf |> rb |> pca)+(numf |> rb |> fa)+(numf |> rb |> ica)+(catf |> ohe )) |> lsvc
