module Pipelines

using DataFrames
using Random

using AutoMLPipeline.AbsTypes
using AutoMLPipeline.BaseFilters
using AutoMLPipeline.Utils
using AutoMLPipeline.EnsembleMethods: BestLearner

import AutoMLPipeline.AbsTypes: fit!, transform!
export fit!, transform!
export Pipeline, ComboPipeline, @pipeline, @pipelinex,processexpr!,sympipeline

"""
    Pipeline(machs::Vector{<:Machine},args::Dict=Dict())

Linear pipeline which iteratively calls and passes the result
of `fit_transform` to the succeeding elements in the pipeline.

Implements `fit!` and `transform!`.
"""
mutable struct Pipeline <: Workflow
  name::String
  model::Dict
  args::Dict

  function Pipeline(args::Dict = Dict())
    default_args = Dict(
			:name => "linearpipeline",
			# machines as list to chain in sequence.
			:machines => Vector{Machine}(),
			# Transformer args as list applied to same index transformer.
			:machine_args => Dict()
		       )
    cargs = nested_dict_merge(default_args, args)
    cargs[:name] = cargs[:name]*"_"*randstring(3)
    new(cargs[:name],Dict(),cargs)
  end
end

"""
    Pipeline(machs::Vector{<:Machine},args::Dict=Dict())

Helper function for Pipeline structure.
"""
function Pipeline(machs::Vector{<:Machine},args::Dict=Dict())
  Pipeline(Dict(:machines => machs, args...))
end


"""
    Pipeline(machs::Vararg{Machine})

Helper function for Pipeline structure.
"""
function Pipeline(machs::Vararg{Machine})
  combo=nothing
  if eltype(machs) <: Machine
    v=[x for x in machs] # convert tuples to vector
    combo = Pipeline(v)
  else
    error("argument setup error")
  end
  return combo
end

function fit!(pipe::Pipeline, features::DataFrame=DataFrame(), labels::Vector=[])
  instances=deepcopy(features)
  machines = pipe.args[:machines]
  machine_args = pipe.args[:machine_args]

  current_instances = instances
  new_machines = Machine[]

  # fit-transform all except last element
  # last element calls fit only
  trlength = length(machines)
  for t_index in 1:(trlength - 1)
    machine = createmachine(machines[t_index], machine_args)
    push!(new_machines, machine)
    fit!(machine, current_instances, labels)
    current_instances = transform!(machine, current_instances)
  end
  machine = createmachine(machines[trlength], machine_args)
  push!(new_machines, machine)
  fit!(machine, current_instances, labels)

  pipe.model = Dict(
      :machines => new_machines,
      :machine_args => machine_args
  )
end

function transform!(pipe::Pipeline, instances::DataFrame=DataFrame())
  machines = pipe.model[:machines]

  current_instances = deepcopy(instances)
  for t_index in 1:length(machines)
    machine = machines[t_index]
    current_instances = transform!(machine, current_instances)
  end

  return current_instances
end


"""
    ComboPipeline(machs::Vector{T}) where {T<:Machine}

Feature union pipeline which iteratively calls 
`fit_transform` of each element and concatenate
their output into one dataframe.

Implements `fit!` and `transform!`.
"""
mutable struct ComboPipeline <: Workflow
  name::String
  model::Dict
  args::Dict

  function ComboPipeline(args::Dict = Dict())
    default_args = Dict(
	:name => "combopipeline",
	# machines as list to chain in sequence.
	:machines => Vector{Machine}(),
	# Transformer args as list applied to same index transformer.
	:machine_args => Dict()
       )
    cargs = nested_dict_merge(default_args, args)
    cargs[:name] = cargs[:name]*"_"*randstring(3)
    new(cargs[:name],Dict(),cargs)
  end
end

function ComboPipeline(machs::Vector{<:Machine},args::Dict=Dict()) 
  ComboPipeline(Dict(:machines => machs, args...))
end

function ComboPipeline(machs::Vararg{Machine})
  combo=nothing
  if eltype(machs) <: Machine
    v=[eval(x) for x in machs] # convert tuples to vector
    combo = ComboPipeline(v)
  else
    error("argument setup error")
  end
  return combo
end


function fit!(pipe::ComboPipeline, features::DataFrame, labels::Vector=[])
  instances=deepcopy(features)
  machines = pipe.args[:machines]
  machine_args = pipe.args[:machine_args]

  new_machines = Machine[]
  new_instances = DataFrame()
  trlength = length(machines)
  for t_index in eachindex(machines)
    machine = createmachine(machines[t_index], machine_args)
    push!(new_machines, machine)
    fit!(machine, instances, labels)
  end

  pipe.model = Dict(
      :machines => new_machines,
      :machine_args => machine_args
  )
end

function transform!(pipe::ComboPipeline, features::DataFrame=DataFrame())
  machines = pipe.model[:machines]
  instances = deepcopy(features)
  new_instances = DataFrame()
  for t_index in eachindex(machines)
    machine = machines[t_index]
    current_instances = transform!(machine, instances)
    new_instances = hcat(new_instances,current_instances,makeunique=true)
  end

  return new_instances
end

function processexpr!(args::AbstractVector)
  for ndx in eachindex(args)
    if typeof(args[ndx]) == Expr
      processexpr!(args[ndx].args)
    elseif args[ndx] == :(|>)
      args[ndx] = :Pipeline
    elseif args[ndx] == :+
      args[ndx] = :ComboPipeline
    elseif args[ndx] == :*
      args[ndx] = :BestLearner
    end
  end
  return nothing
end

# check if quoted expression 
macro pipeline(expr)
  lexpr = :($(esc(expr)))
  if expr isa Expr && expr.head === :quote
    lexpr = :($(esc(expr.args[1])))
  end
  processexpr!(lexpr.args)
  #lexpr.args = res
  lexpr
end

macro pipelinex(expr)
  lexpr = :($(esc(expr)))
  if expr isa Expr && expr.head === :quote
    lexpr = :($(esc(expr.args[1])))
  end
  processexpr!(lexpr.args)
  :($(lexpr.args[1]))
end

function sympipeline(expr)
  processexpr!(expr.args)
  expr
end

end
