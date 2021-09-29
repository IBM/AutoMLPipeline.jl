# ----------------------------------
# Implementation
# ----------------------------------

using Distributed
nprocs() == 1 && addprocs(; exeflags = "--project")
workers()

@everywhere using AutoMLPipeline
@everywhere using DataFrames
@everywhere using AutoMLPipeline: score
@everywhere using Random
ENV["COLUMNS"]=1000
ENV["LINES"]=100;

# disable truncation of dataframes columns
import Base.show
show(df::AbstractDataFrame) = show(df,truncate=0)
show(io::IO,df::AbstractDataFrame) = show(io,df;truncate=0)

# brief overview of AutoMLPipeline package
# more details is in my Juliacon2020 presentation about this package
# we use AutoMLPipeline to develop the pipeline optimizer

# define scalers
rb     = SKPreprocessor("RobustScaler",Dict(:name=>"rb"))
pt     = SKPreprocessor("PowerTransformer",Dict(:name=>"pt"))
norm   = SKPreprocessor("Normalizer",Dict(:name=>"norm"))
mx     = SKPreprocessor("MinMaxScaler",Dict(:name=>"mx"))
std    = SKPreprocessor("StandardScaler",Dict(:name=>"std"))
# define extractors
pca    = SKPreprocessor("PCA",Dict(:name=>"pca"))
fa     = SKPreprocessor("FactorAnalysis",Dict(:name=>"fa"))
ica    = SKPreprocessor("FastICA",Dict(:name=>"ica"))
# define learners
rf     = SKLearner("RandomForestClassifier",Dict(:name => "rf"))
ada    = SKLearner("AdaBoostClassifier",Dict(:name => "ada"))
gb     = SKLearner("GradientBoostingClassifier",Dict(:name => "gb"))
lsvc   = SKLearner("LinearSVC",Dict(:name => "lsvc"))
rbfsvc = SKLearner("SVC",Dict(:name => "rbfsvc"))
dt     = SKLearner("DecisionTreeClassifier",Dict(:name =>"dt"))
# preprocessing
noop = Identity(Dict(:name =>"noop"))
ohe  = OneHotEncoder(Dict(:name=>"ohe"))
catf = CatFeatureSelector(Dict(:name=>"catf"))
numf = NumFeatureSelector(Dict(:name=>"numf"))

vscalers =    [rb,pt,norm,mx,std,noop]
vextractors = [pca,fa,ica,noop]
vlearners =   [rf,gb,lsvc,rbfsvc,ada,dt]

dataset = getiris()
X = dataset[:,1:4]
Y = dataset[:,5] |> collect

# one-block pipeline
prep = @pipeline (catf |> ohe) + numf
ppl1 = @pipeline prep |> (mx |> pca) |> rf
crossvalidate(ppl1,X,Y)

# two-block pipeline
ppl2 = @pipeline prep |> ((norm |> ica) + (pt |> pca)) |> lsvc
crossvalidate(ppl2,X,Y)

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
res_oneblock = oneblock_pipeline_factory(vscalers,vextractors,vlearners)

crossvalidate(res_oneblock.Pipeline[1],X,Y)

function evaluate_pipeline(dfpipelines,X,Y;folds=3)
   res=@distributed (vcat) for prow in eachrow(dfpipelines)
      perf = crossvalidate(prow.Pipeline,X,Y;nfolds=folds)
      DataFrame(Description=prow.Description,mean=perf.mean,sd=perf.std,Pipeline=prow.Pipeline,)
   end
   return res
end

cross_res_oneblock = evaluate_pipeline(res_oneblock,X,Y)
sort!(cross_res_oneblock,:mean, rev = true)
show(cross_res_oneblock;allrows=true,allcols=true,truncate=0)

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
res_twoblock = twoblock_pipeline_factory(vscalers,vextractors,vlearners)

function model_selection_pipeline(learners)
   results = @distributed (vcat) for lr in learners
      prep = @pipeline ((catf |> ohe) + numf)
      expx = @pipeline prep |> (rb |> pca)  |> lr
      pname = "(rb |> pca) |> $(lr.name[1:end-4])"
      DataFrame(Description=pname,Pipeline=expx)
   end
   return results
end
dfpipes = model_selection_pipeline(vlearners)

# find the best model
@everywhere Random.seed!(10)
res = evaluate_pipeline(dfpipes,X,Y;folds=10)
sort!(res,:mean, rev = true)

# use the best model to generate pipeline search
dfp = twoblock_pipeline_factory(vscalers,vextractors,[rbfsvc])

# evaluate the pipeline
@everywhere Random.seed!(10)
bestp=evaluate_pipeline(dfp,X,Y;folds=3)
sort!(bestp,:mean, rev = true)
show(bestp;allrows=false,truncate=0,allcols=false)

# create pipeline factory
#   - create oneblock factory
#   - create twoblocks factory
#
# evaluate pipeline factory

# Model selection should be given more priority than pipeline and hyperparam search
# Algo:
#   - start with an arbitrary pipeline and search for the best model 
#     using their default parameters
#   - with the best model selected, refine the solution 
#     by searching for the best pipeline
#   - refine the model more by hyperparm search if needed
