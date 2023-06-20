using Distributed

nprocs() == 1 && addprocs() 

@everywhere begin
   using AutoOfflineRL
   using AutoMLPipeline
   using Parquet
   using DataFrames
end
@everywhere begin
   # load preprocessing elements
   #### Scaler
   rb = SKPreprocessor("RobustScaler");
   pt = SKPreprocessor("PowerTransformer");
   norm = SKPreprocessor("Normalizer");
   mx = SKPreprocessor("MinMaxScaler");
   std = SKPreprocessor("StandardScaler")
   ##### Column selector
   catf = CatFeatureSelector();
   numf = NumFeatureSelector();
   ## load filters
   ##### Decomposition
   #apca = SKPreprocessor("PCA",Dict(:autocomponent=>true,:name=>"autoPCA"));
   #afa = SKPreprocessor("FactorAnalysis",Dict(:autocomponent=>true,:name=>"autoFA"));
   #aica = SKPreprocessor("FastICA",Dict(:autocomponent=>true,:name=>"autoICA"));
   pca = SKPreprocessor("PCA");
   fa = SKPreprocessor("FactorAnalysis");
   ica = SKPreprocessor("FastICA");
   noop = Identity(Dict(:name => "Noop"));
end

# load dataset
path = pkgdir(AutoOfflineRL)
dataset = "$path/data/smalldata.parquet"
df = Parquet.read_parquet(dataset) |> DataFrame |> dropmissing

df_input = df[:, ["day", "hour", "minute", "dow", "metric1", "metric2", "metric3", "metric4"]]
reward = df[:,["reward"]] |> deepcopy |> DataFrame
action = df[:,["action"]] |> deepcopy |> DataFrame
action_reward = DataFrame[action, reward]

agent = DiscreteRLOffline("NFQ")
pipe = (numf |> mx |> pca) |> agent
crossvalidateRL(pipe,df_input,action_reward)

function pipelinesearch()
   agentnames = ["DiscreteCQL","NFQ","DoubleDQN","DiscreteSAC","DiscreteBCQ","DiscreteBC","DQN"]
   scalers =  [rb,pt,norm,std,mx,noop]
   extractors = [pca,ica,fa,noop]
   dfresults = @sync @distributed (vcat) for agentname in agentnames
      @distributed (vcat) for sc in scalers
         @distributed (vcat) for xt  in extractors
            try
               rlagent = DiscreteRLOffline(agentname,Dict(:runtime_args=>Dict(:n_epochs=>1)))
               rlpipeline = ((numf |> sc |> xt)) |> rlagent 
               res = crossvalidateRL(rlpipeline,df_input,action_reward)
               scn   = sc.name[1:end - 4]; xtn = xt.name[1:end - 4]; lrn = rlagent.name[1:end - 4]
               pname = "$scn |> $xtn |> $lrn"
               if !isnan(res)
                  DataFrame(pipeline=pname,perf=res)
               else
                  DataFrame()
               end
            catch e
               println("error in $agentname")
               DataFrame()
            end
         end
      end
   end
   sort!(dfresults,:perf,rev=false)
   return dfresults
end
dftable= pipelinesearch()
show(dftable,allcols=true,allrows=true,truncate=0)

