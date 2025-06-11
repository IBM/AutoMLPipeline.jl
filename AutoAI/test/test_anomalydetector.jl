module TestAnomalyDetection
using Test
using CSV
using AutoAI
using DataFrames: DataFrame

# anomaly detection
function detect()
    df = CSV.File("./iris.csv") |> DataFrame
    X            = df[:, 1:end-1]
    iforest      = SKAnomalyDetector("IsolationForest")
    ellenv       = SKAnomalyDetector("EllipticEnvelope")
    osvm         = SKAnomalyDetector("OneClassSVM")
    lcfactor     = SKAnomalyDetector("LocalOutlierFactor")
    res_iforest  = fit_transform!(iforest,X)
    res_ellenv   = fit_transform!(ellenv,X)
    res_osvm     = fit_transform!(osvm,X)
    res_lcfactor = fit_transform!(lcfactor,X)
    return DataFrame(iForest=res_iforest,
        ellEnv=res_ellenv,
        oSVM=res_osvm,
        lcFactor=res_lcfactor
    )
end

@testset "anomaly detection" begin
    df = detect()
    @test df |> Matrix .|> abs |> sum == 600
end

end

