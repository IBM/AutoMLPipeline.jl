function initmlflowcargs!(cargs::Dict)
  experiment_tags = Dict(
    "projectname" => cargs[:projectname],
    "projecttype" => cargs[:projecttype],
    "notes" => cargs[:description]
  )
  # check if mlflow server exists
  try
    httpget = getproperty(REQ, "get")
    res = httpget(cargs[:url] * "/health")
  catch
    @error("Mlflow Server Unreachable")
    exit(1)
  end
  MLF.set_tracking_uri(uri=cargs[:url])
  name = cargs[:name]
  experiment = MLF.search_experiments(filter_string="name = \'$name\'")
  if PYC.pylen(experiment) != 0
    MLF.set_experiment(experiment[0].name)
  else
    theexperiment = MLF.create_experiment(name=name, tags=experiment_tags)
    cargs[:experiment_id] = theexperiment
  end
end

function setupautofit!(mlf::Workflow)
  MLF.end_run()
  # generate run name
  run_name = mlf.model[:name] * "_" * "fit" * "_" * randstring(3)
  mlf.model[:run_name] = run_name
  MLF.set_experiment(mlf.model[:name])
  MLF.start_run(run_name=run_name)
  # get run_id
  run = MLF.active_run()
  mlf.model[:run_id] = run.info.run_id
end

function logmlartifact(mlf::Workflow)
  # save model in mlflow
  artifact_name = mlf.model[:artifact_name]
  # use temporary directory
  tmpdir = tempdir()
  artifact_location = joinpath(tmpdir, artifact_name)
  automodel = mlf.model[:automodel]
  serialize(artifact_location, automodel)
  MLF.log_artifact(artifact_location)
  bestmodel_uri = MLF.get_artifact_uri(artifact_path=artifact_name)
  # save model  uri location
  mlf.model[:bestmodel_uri] = bestmodel_uri
  MLF.end_run()
end

function autotransform!(mlf::Workflow, X::DataFrame)
  MLF.end_run()
  # download model artifact
  run_id = mlf.model[:run_id]
  artifact_name = mlf.model[:artifact_name]
  try
    model_artifacts = MLF.artifacts.list_artifacts(run_id=run_id)
    @assert model_artifacts[0].path |> string == artifact_name
  catch e
    @info e
    throw("Artifact $artifact_name does not exist in run_id = $run_id")
  end
  run_name = mlf.model[:name] * "_" * "transform" * "_" * randstring(3)
  mlf.model[:run_name] = run_name
  MLF.set_experiment(mlf.model[:name])
  MLF.start_run(run_name=run_name)
  pylocalpath = MLF.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_name)
  bestmodel = deserialize(string(pylocalpath))
  Y = transform!(bestmodel, X)
  MLF.log_param("output", Y)
  MLF.end_run()
  return Y
end
