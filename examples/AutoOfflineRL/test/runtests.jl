module TestOfflineRL
using AutoOfflineRL
using Test
using DataFrames
using PythonCall
using Parquet
const PYC=PythonCall

@testset "Load Agents with Default Params" begin
  for agentid in keys(AutoOfflineRL.OfflineRLs.rl_dict)
    @info "loading $agentid default params"
    rlagent = DiscreteRLOffline(agentid) 
    @test typeof(rlagent) <: AutoOfflineRL.Learner
  end
end


@testset "Load Agents with Param Args" begin
  println()
  for agentid in keys(AutoOfflineRL.OfflineRLs.rl_dict)
    @info "loading $agentid with customized params"
    rlagent = DiscreteRLOffline(agentid,
          Dict(:name=>agentid,
               :iterations=>10000,
               :epochs=>100)
         ) 
    @test typeof(rlagent) <: AutoOfflineRL.Learner
  end
end

@testset "Test Exceptions" begin
  @test_throws ErrorException DiscreteRLOffline("dummy")
end

@testset "Test Agent fit!/transform Runs" begin
  println()
  path = pkgdir(AutoOfflineRL)
  dataset = "$path/data/smalldata.parquet"
  df = Parquet.read_parquet(dataset) |> DataFrame |> dropmissing
  for agentid in keys(AutoOfflineRL.OfflineRLs.rl_dict)
    @info "training $agentid"
    agent = DiscreteRLOffline(agentid; save_model=false)
    o_header = agent.model[:o_header]
    fit!(agent,df; n_epochs=1,verbose=false, show_progress=true) 
    @test agent.model[:rlobjtrained] !== PYC.PyNULL
    @info "transform $agentid"
    adf = df[1:2,:]
    if agentid != "DiscreteBC"
       res = AutoOfflineRL.transform!(agent,adf)
       @test typeof(res[1]) .== NamedTuple{(:obs,:action, :value), Tuple{Vector{Float64},Vector{Float64}, Vector{Float64}}}
    end
  end
end




end
