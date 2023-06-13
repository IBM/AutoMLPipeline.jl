module OfflineRLs

using AutoOfflineRL
using Parquet
using Distributed
using DataFrames: DataFrame, dropmissing
using Random
using CSV
using Dates

using ..AbsTypes
using ..Utils
import ..AbsTypes: fit, fit!, transform, transform!

import PythonCall
const PYC = PythonCall

using ..Utils: nested_dict_merge

export DiscreteRLOffline, fit!, transform!, fit, transform
export listdiscreateagents, driver

const rl_dict = Dict{String, PYC.Py}()

const PYRL = PYC.pynew()
const PYPD = PYC.pynew()
const PYNP = PYC.pynew()
const PYDT = PYC.pynew()

function __init__()
  PYC.pycopy!(PYRL, PYC.pyimport("d3rlpy.algos"))
  PYC.pycopy!(PYDT, PYC.pyimport("d3rlpy.dataset"))
  PYC.pycopy!(PYPD, PYC.pyimport("pandas"))
  PYC.pycopy!(PYNP, PYC.pyimport("numpy"))

  # OfflineRLs
  rl_dict["DiscreteBC"]           = PYRL
  rl_dict["DQN"]                  = PYRL
  rl_dict["NFQ"]                  = PYRL
  rl_dict["DoubleDQN"]            = PYRL
  rl_dict["DiscreteBCQ"]          = PYRL
  rl_dict["DiscreteCQL"]          = PYRL
  rl_dict["DiscreteSAC"]          = PYRL
  #rl_dict["DiscreteRandomPolicy"] = PYRL
end

mutable struct DiscreteRLOffline <: Learner
   name::String
   model::Dict{Symbol,Any}

   function DiscreteRLOffline(args=Dict{Symbol,Any}())
     default_args = Dict{Symbol,Any}(
        :name         => "DQN",
        :tag          => "RLOffline",
        :rlagent      => "DQN",
        :iterations   => 100,
        :save_metrics => false,
        :rlobjtrained => PYC.PyNULL,
        :o_header     => ["day", "hour", "minute", "dow", "metric1", "metric2", "metric3", "metric4"],
        :a_header     => ["action"],
        :r_header     => ["reward"],
        :save_model   => false,
        :runtime_args => Dict{Symbol, Any}(
            :n_epochs => 5,
        ),
        :impl_args    => Dict{Symbol,Any}(
            :scaler  => "min_max",
            :use_gpu => false,
        )
     )
     cargs = nested_dict_merge(default_args,args)
     datestring = Dates.format(now(), "yyyy-mm-dd-HH-MM")
     cargs[:name] = cargs[:name]*"_"*datestring
     rlagent = cargs[:rlagent]
     if !(rlagent in keys(rl_dict)) 
       println("error: $rlagent is not supported.")
       println()
       discreteagents()
       error("Argument keyword error")
     end
     new(cargs[:name],cargs)
   end
end

function DiscreteRLOffline(rlagent::String, args::Dict)
  DiscreteRLOffline(Dict(:rlagent => rlagent, :name => rlagent, args...))
end

function DiscreteRLOffline(rlagent::String; args...)
  DiscreteRLOffline(Dict(
          :rlagent => rlagent, 
          :name => rlagent, 
          args...
  ))
end

function listdiscreateagents()
  println()
  println("RL Discrete Agents:")
  agents = keys(rl_dict) |> collect
  [println("  ",agent," ") for agent in agents]
  println("See d3rlpy python package for details about the agent arguments.")
  nothing
end

function discreteagents()
  println()
  println("syntax: DiscreteRLOffline(name::String, args::Dict)")
  println("and 'args' are the agent's parameters")
  println("See d3rlpy python package for details about the agent arguments.")
  println("use: listdiscreateagents() to get the available RL agents")
end

function createmdpdata!(agent::DiscreteRLOffline, df::DataFrame, action_reward::Vector)
  _observations = df |> Array .|> PYC.float |> x -> PYNP.array(x, dtype = "float32")
  _actions      = action_reward[1] |> Array .|> PYC.float |> x -> PYNP.array(x, dtype = "float32")
  _rewards      = action_reward[2] |> Array .|> PYC.float |> x -> PYNP.array(x, dtype = "float32")
  ## inject end of data by terminal column
  nrow, _         = size(df)
  _terminals      = zeros(Int, nrow)
  _terminals[end] = 1
  _terminals      = _terminals .|> PYC.float |> x -> PYNP.array(x, dtype = "int32")
  ## create dataset for RLOffline
  mdp_dataset = PYDT.MDPDataset(
    observations = _observations,
    actions      = _actions,
    rewards      = _rewards,
    terminals    = _terminals,
  )
  ## save params
  agent.model[:mdp_dataset]     = mdp_dataset
  agent.model[:np_observations] = _observations
  agent.model[:np_actions]      = _actions
  agent.model[:np_rewards]      = _rewards
  return mdp_dataset
end

function checkheaders(agent::DiscreteRLOffline, df)
  o_header    = agent.model[:o_header]  
  a_header    = agent.model[:a_header]
  r_header    = agent.model[:r_header]
  dfnames = names(df)
  [@assert header in dfnames "\"$header\" is not in data header" 
   for header in vcat(o_header, a_header, r_header)]
end

function fit!(agent::DiscreteRLOffline, df::DataFrame, action_reward::Vector)::Nothing
  # check if headers exist
  #checkheaders(agent::DiscreteRLOffline, df)
  # create mdp data
  nrow, ncol  = size(df)
  mdp_dataset = createmdpdata!(agent, df,action_reward)
  ## prepare algorithm
  runtime_args = agent.model[:runtime_args]
  logging    = agent.model[:save_metrics]
  impl_args  = copy(agent.model[:impl_args])
  rlagent    = agent.model[:rlagent]
  py_rlagent = getproperty(rl_dict[rlagent],rlagent)
  pyrlobj    = py_rlagent(;impl_args...)
  pyrlobj.fit(mdp_dataset; save_metrics = logging, runtime_args... )
  ## save rl to model dictionary
  agent.model[:rlobjtrained] = pyrlobj
  agent.model[:nrow]         = nrow
  agent.model[:ncol]         = ncol
  ## save model to file
  if agent.model[:save_model] == true
    path      = pkgdir(AutoOfflineRL)
    agentname = agent.model[:name]
    tag       = agent.model[:tag]
    fnmodel     = "$path/model/$(agentname)_$(tag)_model.pt"
    fnpolicy  = "$path/model/$(agentname)_$(tag)_policy.pt"
    pyrlobj.save_model(fnmodel)
    pyrlobj.save_policy(fnpolicy)
  end
  return nothing
end


function transform!(agent::DiscreteRLOffline,df::DataFrame=DataFrame())::Vector
  pyrlobj  = agent.model[:rlobjtrained]
  #o_header = agent.model[:o_header]
  observations = df |> Array .|> PYC.float |> x -> PYNP.array(x, dtype = "float32")
  res = map(observations) do obs
     action   = pyrlobj.predict(obs) 
     value    = pyrlobj.predict_value([obs],action) 
     action = PYC.pyconvert.(Float64,action)
     value = PYC.pyconvert.(Float64,value)
     obs = PYC.pyconvert.(Float64,obs)
     (;obs,action,value)
  end
  return res
end

function driver()
  #dataset = ENV["HOME"]*"/phome/ibmgithub/ZOS/data/processed_batch.csv"
  path = pkgdir(AutoOfflineRL)
  dataset = "$path/data/smalldata.parquet"
  df = Parquet.read_parquet(dataset) |> DataFrame |> dropmissing
  #for agentid in reverse([keys(rl_dict)...])
  #  println(agentid)
  #  if agentid != "DiscreteRandomPolicy"
  #    agent = DiscreteRLOffline(agentid)
  #    fit!(agent,df)
  #  end
  #end
  #discreteagents()
  agent = DiscreteRLOffline("DoubleDQN"; tag="sac")
  header = agent.model[:o_header]
  fit!(agent,df)
  transform!(agent,df[1:20,:])
  #vec = df[1,header] |> Vector
  #transform(agent,vec)
  #m = DiscreteRLOffline()
  #dqn_agent(m,dat)
end


end
