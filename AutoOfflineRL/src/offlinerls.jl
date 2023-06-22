module OfflineRLs

using AutoOfflineRL
using PythonCall
import Statistics
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
export crossvalidateRL

const rl_dict = Dict{String, PYC.Py}()
const metric_dict = Dict{String, PYC.Py}()

const PYRL = PYC.pynew()
const PYPD = PYC.pynew()
const PYNP = PYC.pynew()
const PYDT = PYC.pynew()
const PYMT = PYC.pynew()
const PYSK  = PYC.pynew()


function __init__()
  PYC.pycopy!(PYRL, PYC.pyimport("d3rlpy.algos"))
  PYC.pycopy!(PYDT, PYC.pyimport("d3rlpy.datasets"))
  PYC.pycopy!(PYSK, PYC.pyimport("sklearn.model_selection"))
  PYC.pycopy!(PYMT, PYC.pyimport("d3rlpy.metrics"))
  PYC.pycopy!(PYPD, PYC.pyimport("pandas"))
  PYC.pycopy!(PYNP, PYC.pyimport("numpy"))

  # OfflineRLs
  metric_dict["cross_validate"]   = PYSK.cross_validate
  metric_dict["train_test_split"] = PYSK.train_test_split
  metric_dict["td_error_scorer"]  = PYMT.td_error_scorer
  metric_dict["discrete_action_match_scorer"] = PYMT.discrete_action_match_scorer
  metric_dict["average_value_estimation_scorer"]  = PYMT.average_value_estimation_scorer
  metric_dict["get_cartpole"] = PYDT.get_cartpole

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
            :n_epochs => 3,
        ),
        :impl_args    => Dict{Symbol,Any}(
            :scaler  => "min_max",
            :use_gpu => false,
        )
     )
     cargs = nested_dict_merge(default_args,args)
     #datestring = Dates.format(now(), "yyyy-mm-dd-HH-MM")
     cargs[:name] = cargs[:name]*"_"*randstring(3)
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

function createmdpdata!(agent::DiscreteRLOffline, df::DataFrame, action_reward_term::Vector)
  _observations = df |> Array .|> PYC.float |> x -> PYNP.array(x, dtype = "float32")
  _actions      = action_reward_term[1] |> Array .|> PYC.float |> x -> PYNP.array(x, dtype = "float32")
  _rewards      = action_reward_term[2] |> Array .|> PYC.float |> x -> PYNP.array(x, dtype = "float32")
  _terminals      = action_reward_term[3] |> Array .|> PYC.float |> x -> PYNP.array(x, dtype = "float32")
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

function fit!(agent::DiscreteRLOffline, df::DataFrame, action_reward_term::Vector)::Nothing
  # check if headers exist
  #checkheaders(agent::DiscreteRLOffline, df)
  # create mdp data
  nrow, ncol  = size(df)
  mdp_dataset = createmdpdata!(agent, df,action_reward_term)
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

function prp_fit_transform(pipe::Machine, instances::DataFrame,actrewterm::Vector)
   machines = pipe.model[:machines]
   machine_args = pipe.model[:machine_args]

   current_instances = instances
   trlength = length(machines)
   for t_index in 1:(trlength - 1)
      machine = createmachine(machines[t_index], machine_args)
      fit!(machine, current_instances, actrewterm)
      current_instances = transform!(machine, current_instances)
   end
   return current_instances
end


function driver()
  path = pkgdir(AutoOfflineRL)
  dataset = "$path/data/smalldata.parquet"
  df = Parquet.read_parquet(dataset) |> DataFrame |> dropmissing
  df_input = df[:, ["day", "hour", "minute", "dow", "metric1", "metric2", "metric3", "metric4"]]
  reward = df[:,["reward"]] |> deepcopy |> DataFrame
  action = df[:,["action"]] |> deepcopy |> DataFrame
  action_reward = DataFrame[action, reward]
  agentname="NFQ"
  agent = DiscreteRLOffline(agentname)
  #fit_transform!(agent,df_input,action_reward)
end

function traintesteval(agent::DiscreteRLOffline,mdp_dataset::Py)
   runtime_args = agent.model[:runtime_args]
   logging    = agent.model[:save_metrics]
   impl_args  = copy(agent.model[:impl_args])
   rlagent    = agent.model[:rlagent]
   py_rlagent = getproperty(rl_dict[rlagent],rlagent)
   pyrlobj    = py_rlagent(;impl_args...)
   py_train_test_split = metric_dict["train_test_split"]
   trainepisodes,testepisodes = py_train_test_split(mdp_dataset)

   td_error_scorer  = PYMT.td_error_scorer
   discrete_action_match_scorer  = PYMT.discrete_action_match_scorer
   runconfig = Dict(:scorers=>Dict("td_error"=>td_error_scorer))
   #runconfig = Dict(:scorers=>Dict("metric"=>discrete_action_match_scorer))
   score=pyrlobj.fit(trainepisodes;
                     eval_episodes=testepisodes,
                     runtime_args...,runconfig...)
   vals = pyconvert(Array,score)
   mvals = [v[2]["td_error"] for v in vals] |> Statistics.mean
   #mvals = [v[2]["metric"] for v in vals] |> Statistics.mean
   return mvals
end

function crossvalidateRL(pp::Machine, dfobs::DataFrame, actrewterm::Vector; cv=3)
   pipe = deepcopy(pp)
   features = deepcopy(dfobs)
   machines = pipe.model[:machines]
   agent = machines[end]
   
   df_input = prp_fit_transform(pipe,features,actrewterm)
   mdp_dataset = createmdpdata!(agent,df_input,actrewterm)

   scores= [traintesteval(agent,mdp_dataset) for i in 1:cv]
   return Statistics.mean(scores)

   #pyskcrossvalidate = metric_dict["cross_validate"]  
   #td_error_scorer  = metric_dict["td_error_scorer"]
   #average_value_estimation_scorer = metric_dict["average_value_estimation_scorer"]
   #runconfig = Dict(:scoring=>Dict("td_error"=>td_error_scorer,
   #                                "value_scale"=>average_value_estimation_scorer),
   #                 :fit_params=>Dict("n_epochs"=>1))
   #scores = pyskcrossvalidate(pyrlobj,mdp_dataset; runconfig...)
   #return scores
end


end
