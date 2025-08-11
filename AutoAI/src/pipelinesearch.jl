# define scalers
const rb = SKPreprocessor("RobustScaler", Dict(:name => "rb"))
const pt = SKPreprocessor("PowerTransformer", Dict(:name => "pt"))
const norm = SKPreprocessor("Normalizer", Dict(:name => "norm"))
const mx = SKPreprocessor("MinMaxScaler", Dict(:name => "mx"))
const std = SKPreprocessor("StandardScaler", Dict(:name => "std"))
# define extractors
const pca = SKPreprocessor("PCA", Dict(:name => "pca"))
const fa = SKPreprocessor("FactorAnalysis", Dict(:name => "fa"))
const ica = SKPreprocessor("FastICA", Dict(:name => "ica"))
# preprocessing
const noop = Identity(Dict(:name => "noop"))
const ohe = OneHotEncoder(Dict(:name => "ohe"))
const catf = CatFeatureSelector(Dict(:name => "catf"))
const numf = NumFeatureSelector(Dict(:name => "numf"))

const _gscalersdict = Dict("rb" => rb, "pt" => pt,
  "norm" => norm, "mx" => mx,
  "std" => std, "noop" => noop)
const _gextractordict = Dict("pca" => pca, "fa" => fa,
  "ica" => ica, "noop" => noop)


function evaluate_pipeline(wflow::Workflow, X::DataFrame, Y::Vector)
  dfpipelines = wflow.model[:dfpipelines]
  folds = wflow.model[:nfolds]
  pmetric = wflow.model[:metric]
  res = @distributed (vcat) for prow in eachrow(dfpipelines)
    perf = crossvalidate(prow.Pipeline, X, Y, pmetric; nfolds=folds)
    DataFrame(; Description=prow.Description, mean=perf.mean, sd=perf.std, prow.Pipeline)
  end
  return res
end

function oneblock_pipeline_factory(wflow::Workflow)
  learners = [wflow.model[:bestlearner]]
  extractors = wflow.model[:oextractors]
  scalers = wflow.model[:oscalers]
  results = @distributed (vcat) for lr in learners
    @distributed (vcat) for xt in extractors
      @distributed (vcat) for sc in scalers
        # baseline preprocessing
        prep = @pipeline ((catf |> ohe) + numf)
        # one-block prp
        expx = @pipeline prep |> (sc |> xt) |> lr
        scn = sc.name[1:end-4]
        xtn = xt.name[1:end-4]
        lrn = lr.name[1:end-4]
        pname = "($scn |> $xtn) |> $lrn"
        DataFrame(Description=pname, Pipeline=expx)
      end
    end
  end
  return results
end

function twoblock_pipeline_factory(wflow::Workflow)
  learners = [wflow.model[:bestlearner]]
  extractors = wflow.model[:oextractors]
  scalers = wflow.model[:oscalers]
  results = @distributed (vcat) for lr in learners
    @distributed (vcat) for xt1 in extractors
      @distributed (vcat) for xt2 in extractors
        @distributed (vcat) for sc1 in scalers
          @distributed (vcat) for sc2 in scalers
            prep = @pipeline ((catf |> ohe) + numf)
            expx = @pipeline prep |> ((sc1 |> xt1) + (sc2 |> xt2)) |> lr
            scn1 = sc1.name[1:end-4]
            xtn1 = xt1.name[1:end-4]
            scn2 = sc2.name[1:end-4]
            xtn2 = xt2.name[1:end-4]
            lrn = lr.name[1:end-4]
            pname = "($scn1 |> $xtn1) + ($scn2 |> $xtn2) |> $lrn"
            DataFrame(Description=pname, Pipeline=expx)
          end
        end
      end
    end
  end
  return results
end

function model_selection_pipeline(wflow::Workflow)
  learners = wflow.model[:olearners]
  results = @distributed (vcat) for lr in learners
    prep = @pipeline ((catf |> ohe) + numf)
    expx = @pipeline prep |> (rb |> pca) |> lr
    pname = "(rb |> pca) |> $(lr.name[1:end-4])"
    DataFrame(Description=pname, Pipeline=expx)
  end
  return results
end

function lname(n::Learner)
  n.name[1:end-4]
end

function twoblocksearch(wflow::Workflow, X::DataFrame, Y::Vector)
  # use the best model to generate pipeline search
  dfpipelines = twoblock_pipeline_factory(wflow)
  wflow.model[:dfpipelines] = dfpipelines
  bestp = evaluate_pipeline(wflow, X, Y)
  sort!(bestp, :mean, rev=wflow.model[:sortrev])
  show(bestp; allrows=false, truncate=1, allcols=false)
  wflow.model[:performance] = bestp
  println()
  optmodel = bestp[1, :]
  return optmodel
end

function oneblocksearch(wflow::Workflow, X::DataFrame, Y::Vector)
  # use the best model to generate pipeline search
  dfpipelines = oneblock_pipeline_factory(wflow)
  wflow.model[:dfpipelines] = dfpipelines
  bestp = evaluate_pipeline(wflow, X, Y)
  sort!(bestp, :mean, rev=wflow.model[:sortrev])
  show(bestp; allrows=false, truncate=1, allcols=false)
  wflow.model[:performance] = bestp
  println()
  optmodel = bestp[1, :]
  return optmodel
end
