using AutoOfflineRL
using AutoMLPipeline
using Parquet
using DataFrames

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

rlagent = DiscreteRLOffline(Dict(:name=>"NFQ",:rlagent=>"NFQ",
    :runtime_args=>Dict(:n_epochs=>10)))


pipeline = (numf |> ica ) + (numf |> pca) |> rlagent

fit_transform!(pipeline,df)
