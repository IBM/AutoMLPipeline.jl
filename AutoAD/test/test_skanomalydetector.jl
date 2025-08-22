module TestSKAnomalyDetection
using Test
using CSV
using AutoAD
using DataFrames: DataFrame

# anomaly detection
function detect()
  df = get_iris()
  X = df[:, 1:end-1]
  iforest = SKAnomalyDetector("IsolationForest")
  ellenv = SKAnomalyDetector("EllipticEnvelope")
  osvm = SKAnomalyDetector("OneClassSVM")
  lcfactor = SKAnomalyDetector("LocalOutlierFactor")
  res_iforest = fit_transform!(iforest, X)
  res_ellenv = fit_transform!(ellenv, X)
  res_osvm = fit_transform!(osvm, X)
  res_lcfactor = fit_transform!(lcfactor, X)
  return DataFrame(iForest=res_iforest,
    ellEnv=res_ellenv,
    oSVM=res_osvm,
    lcFactor=res_lcfactor
  )
end

@testset "anomaly detection" begin
  df = detect()
  @test count(>=(0), Matrix(df)) == 600
end

end

