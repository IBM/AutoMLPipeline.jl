module PipelineBlocksRegression

export twoblockspipelinesearch, oneblockpipelinesearch

using Distributed

using AutoMLPipeline
using DataFrames
using DataFrames:DataFrame
using AutoMLPipeline: score
using Random

# disable truncation of dataframes columns
import Base.show
show(df::AbstractDataFrame) = show(df,truncate=0)
show(io::IO,df::AbstractDataFrame) = show(io,df;truncate=0)

# define scalers
const rb     = SKPreprocessor("RobustScaler",Dict(:name=>"rb"))
const pt     = SKPreprocessor("PowerTransformer",Dict(:name=>"pt"))
const norm   = SKPreprocessor("Normalizer",Dict(:name=>"norm"))
const mx     = SKPreprocessor("MinMaxScaler",Dict(:name=>"mx"))
const std    = SKPreprocessor("StandardScaler",Dict(:name=>"std"))
# define extractors
const pca    = SKPreprocessor("PCA",Dict(:name=>"pca"))
const fa     = SKPreprocessor("FactorAnalysis",Dict(:name=>"fa"))
const ica    = SKPreprocessor("FastICA",Dict(:name=>"ica"))
# define learners
const rf     = SKLearner("RandomForestRegressor",Dict(:name => "rf"))
const ada    = SKLearner("AdaBoostRegressor",Dict(:name => "ada"))
const gb     = SKLearner("GradientBoostingRegressor",Dict(:name => "gb"))
const ridge   = SKLearner("Ridge",Dict(:name => "ridge"))
const svr = SKLearner("SVR",Dict(:name => "svr"))
const dt     = SKLearner("DecisionTreeRegressor",Dict(:name =>"dt"))
# preprocessing
const noop = Identity(Dict(:name =>"noop"))
const ohe  = OneHotEncoder(Dict(:name=>"ohe"))
const catf = CatFeatureSelector(Dict(:name=>"catf"))
const numf = NumFeatureSelector(Dict(:name=>"numf"))

const vscalers =    [rb,pt,norm,mx,std,noop]
const vextractors = [pca,fa,ica,noop]
const vlearners =   [rf,gb,ridge,svr,ada,dt]
const learnerdict = Dict("rf"=>rf,"gb"=>gb,"ridge"=>ridge,"svr"=>svr,"ada"=>ada,"dt"=>dt)


function oneblock_pipeline_factory(scalers,extractors,learners)
   results = @distributed (vcat) for lr in learners
      @distributed (vcat) for xt in extractors
         @distributed (vcat) for sc in scalers
            # baseline preprocessing
            prep = @pipeline ((catf |> ohe) + numf)
            # one-block prp
            expx = @pipeline prep |> (sc |> xt) |> lr
            scn   = sc.name[1:end - 4];xtn = xt.name[1:end - 4]; lrn = lr.name[1:end - 4]
            pname = "($scn |> $xtn) |> $lrn"
            DataFrame(Description=pname,Pipeline=expx)
         end
      end
   end
   return results
end

function evaluate_pipeline(dfpipelines,X,Y;folds=3)
   res=@distributed (vcat) for prow in eachrow(dfpipelines)
      perf = crossvalidate(prow.Pipeline,X,Y,"mean_squared_error";nfolds=folds)
      DataFrame(;Description=prow.Description,mean=perf.mean,sd=perf.std,prow.Pipeline)
   end
   return res
end

function twoblock_pipeline_factory(scalers,extractors,learners)
   results = @distributed (vcat) for lr in learners
      @distributed (vcat) for xt1 in extractors
         @distributed (vcat) for xt2 in extractors
            @distributed (vcat) for sc1 in scalers
               @distributed (vcat) for sc2 in scalers
                  prep = @pipeline ((catf |> ohe) + numf)
                  expx = @pipeline prep |> ((sc1 |> xt1) + (sc2 |> xt2)) |> lr
                  scn1   = sc1.name[1:end - 4];xtn1 = xt1.name[1:end - 4]; 
                  scn2   = sc2.name[1:end - 4];xtn2 = xt2.name[1:end - 4]; 
                  lrn = lr.name[1:end - 4]
                  pname = "($scn1 |> $xtn1) + ($scn2 |> $xtn2) |> $lrn"
                  DataFrame(Description=pname,Pipeline=expx)
               end
            end
         end
      end
   end
   return results
end

function model_selection_pipeline(learners)
   results = @distributed (vcat) for lr in learners
      prep = @pipeline ((catf |> ohe) + numf)
      expx = @pipeline prep |> (rb |> pca)  |> lr
      pname = "(rb |> pca) |> $(lr.name[1:end-4])"
      DataFrame(Description=pname,Pipeline=expx)
   end
   return results
end

function lname(n::Learner)
   n.name[1:end-4]
end

function twoblockspipelinesearch(X::DataFrame,Y::Vector;scalers=vscalers,extractors=vextractors,learners=vlearners,nfolds=3)
   dfpipes = model_selection_pipeline(vlearners)
   # find the best model by evaluating the models
   modelsperf = evaluate_pipeline(dfpipes,X,Y;folds=nfolds)
   sort!(modelsperf,:mean, rev = false)
   # get the string name of the top model
   bestm =  filter(x->occursin(x,modelsperf.Description[1]),lname.(vlearners))[1]
   # get corresponding model object
   bestmodel = learnerdict[bestm]
   # use the best model to generate pipeline search
   dfp = twoblock_pipeline_factory(vscalers,vextractors,[bestmodel])
   # evaluate the pipeline
   bestp=evaluate_pipeline(dfp,X,Y;folds=nfolds)
   sort!(bestp,:mean, rev = false)
   show(bestp;allrows=false,truncate=1,allcols=false)
   println()
   optmodel = bestp[1,:]
   return optmodel
end

function oneblockpipelinesearch(X::DataFrame,Y::Vector;scalers=vscalers,extractors=vextractors,learners=vlearners,nfolds=3)
   dfpipes = model_selection_pipeline(vlearners)
   # find the best model by evaluating the models
   modelsperf = evaluate_pipeline(dfpipes,X,Y;folds=nfolds)
   sort!(modelsperf,:mean, rev = false)
   # get the string name of the top model
   bestm =  filter(x->occursin(x,modelsperf.Description[1]),lname.(vlearners))[1]
   # get corresponding model object
   bestmodel = learnerdict[bestm]
   # use the best model to generate pipeline search
   dfp = oneblock_pipeline_factory(vscalers,vextractors,[bestmodel])
   # evaluate the pipeline
   bestp=evaluate_pipeline(dfp,X,Y;folds=nfolds)
   sort!(bestp,:mean, rev = false)
   show(bestp;allrows=false,truncate=1,allcols=false)
   println()
   optmodel = bestp[1,:]
   return optmodel
end

end
