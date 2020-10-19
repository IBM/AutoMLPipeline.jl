module Baselines

using Random
using DataFrames
using StatsBase: mode

using ..Utils
using ..AbsTypes: Machine, Transformer, Learner, Workflow, Computer

import ..AbsTypes: fit!, transform!

export fit!,transform!
export Baseline, Identity

"""
    Baseline(
       default_args = Dict(
	       :name => "baseline",
          :output => :class,
          :strat => mode,
          :impl_args => Dict()
       )
    )

Baseline model that returns the mode during classification.
"""
mutable struct Baseline <: Learner
   name::String
   model::Dict{Symbol,Any}

   function Baseline(args=Dict())
      default_args = Dict{Symbol,Any}(
         :name      => "baseline",
         :output    => :class,
         :strat     => mode,
         :impl_args => Dict{Symbol,Any}()
      )
      cargs = nested_dict_merge(default_args, args)
      cargs[:name] = cargs[:name]*"_"*randstring(3)
      new(cargs[:name],cargs)
   end
end

"""
    Baseline(name::String,opt...)

 Helper function
"""
function Baseline(name::String;opt...)
   Baseline(Dict(:name=>name,:impl_args=>Dict(pairs(opt))))
end


"""
    fit!(bsl::Baseline,x::DataFrame,y::Vector)

Get the mode of the training data.
"""
function fit!(bsl::Baseline,x::DataFrame,y::Vector)
   bsl.model[:choice] = bsl.model[:strat](y)
end

"""
    transform!(bsl::Baseline,x::DataFrame)

Return the mode in classification.
"""
function transform!(bsl::Baseline,x::DataFrame)
  fill(bsl.model[:choice],size(x,1))
end

"""
    Identity(args=Dict())

Returns the input as output.
"""
mutable struct Identity <: Transformer
  name::String
  model::Dict{Symbol,Any}

  function Identity(args=Dict())
	 default_args = Dict{Symbol,Any}(
				:name => "identity",
				:impl_args => Dict()
			)
	 cargs = nested_dict_merge(default_args, args)
    cargs[:name] = cargs[:name]*"_"*randstring(3)
	 new(cargs[:name],cargs)
  end
end

"""
    Baseline(name::String,opt...)

 Helper function
"""
function Identity(name::String; opt...)
   Baseline(Dict(:name=>name,:impl_args=>Dict(pairs(opt))))
end

"""
    fit!(idy::Identity,x::DataFrame,y::Vector)

Does nothing.
"""
function fit!(idy::Identity,x::DataFrame,y::Vector)
    nothing
end

"""
    transform!(idy::Identity,x::DataFrame)

Return the input as output.
"""
function transform!(idy::Identity,x::DataFrame)
    return x
end

end
