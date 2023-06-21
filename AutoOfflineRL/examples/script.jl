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

#df = df[:,["day", "hour", "minute", "dow"]]
#df.sensor1 = rand(1:500,srow)
#df.sensor2 = rand(1:200,srow)
#df.sensor3 = rand(1:100,srow)
#df.action = rand([10,50,100],srow)
#df.reward = rand(srow)

srow,_ = size(df)
observation = df[:, ["day", "hour", "minute", "dow", "sensor1", "sensor2", "sensor3"]]
reward = df[:,["reward"]] |> deepcopy |> DataFrame
action = df[:,["action"]] |> deepcopy |> DataFrame
_terminals = zeros(Int,srow)
_terminals[collect(100:1000:9000)] .= 1
_terminals[end] = 1
dterminal = DataFrame(terminal=_terminals)
action_reward_terminal = DataFrame[action, reward, dterminal]

agent = DiscreteRLOffline("NFQ")
pipe = (numf |> mx |> pca) |> agent
crossvalidateRL(pipe,observation,action_reward_terminal)

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
               res = crossvalidateRL(rlpipeline,observation,action_reward_terminal)
               scn   = sc.name[1:end - 4]; xtn = xt.name[1:end - 4]; lrn = rlagent.name[1:end - 4]
               pname = "$scn |> $xtn |> $lrn"
               if !isnan(res)
                  DataFrame(pipeline=pname,td_error=res)
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
   #sort!(dfresults,:percent_action_matches,rev=true)
   return dfresults
end
dftable= pipelinesearch()
sort!(dftable,:td_error,rev=false)
show(dftable,allcols=true,allrows=true,truncate=0)
