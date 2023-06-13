using AutoOfflineRL
using AutoMLPipeline
using Parquet
using DataFrames
<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
# load preprocessing elements
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

# load dataset
path = pkgdir(AutoOfflineRL)
dataset = "$path/data/smalldata.parquet"
df = Parquet.read_parquet(dataset) |> DataFrame |> dropmissing

df_input = df[:, ["day", "hour", "minute", "dow", "metric1", "metric2", "metric3", "metric4"]]
reward = df[:,["reward"]] |> deepcopy |> DataFrame
action = df[:,["action"]] |> deepcopy |> DataFrame
action_reward = DataFrame[action, reward]

rlagent = DiscreteRLOffline(Dict(:name=>"NFQ",:rlagent=>"NFQ",
    :runtime_args=>Dict(:n_epochs=>10)))


pipeline = (numf |> ica ) + (numf |> pca) |> rlagent

fit_transform!(pipeline,df)

p = @pipeline ((numf |> ica ) + (numf |> pca)) |> rlagent 

dfnew=fit_transform(p,df_input)

|> rlagent
pipeline = rlagent

fit!(p,df_input,action_reward)

AutoOfflineRL.transform!(pipeline,df_input)



a=fit_transform!(pipeline,df)
