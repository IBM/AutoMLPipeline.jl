module TestAutoAI
using CondaPkg
using Distributed

nprocs() == 1 && addprocs()

@everywhere using AutoAI

include("./test_automl.jl")

#const tmpdir = tempdir()
#const mlflowpath = joinpath(CondaPkg.envdir(), "bin", "mlflow")
#const backendpath = joinpath("sqlite:///", tmpdir, "mlflow.db")
#const artifactpath = joinpath(tmpdir, "mlruns")
#cmd = "$mlflowpath server --host 127.0.0.1 --port 8080 --backend-store-uri $backendpath --default-artifact-root $artifactpath "
#mlflowprocess = run(Cmd(`sh -c "$cmd"`), wait=false)
#
#include("./test_automlflow.jl")
#
#kill(mlflowprocess)

end
