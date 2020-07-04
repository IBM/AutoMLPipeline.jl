module Plotters

using Random
using DataFrames
using Makie, AbstractPlotting, GLMakie

using AutoMLPipeline.AbsTypes
using AutoMLPipeline.BaseFilters
using AutoMLPipeline.Utils

import AutoMLPipeline.AbsTypes: fit!, transform!
export fit!, transform!
export Plotter


"""
    Plotter(
      Dict(
        :name => "plotter",
        :axes => [x, y, ...],
        :type => lines, bar, scatter
      )
    )
Asks for the dimensions and some other arguments
required to make certain type of plot.
Implements `fit!` and `transform!`.
"""
mutable struct Plotter <: Transformer
  name::String
  model::Dict
  args::Dict

  function Plotter(args::Dict = Dict())
	 default_args = Dict(
                          :name => "plotter",                            
                          :axes => Float[]
                        )
         cargs=nested_dict_merge(default_args,args)
	 cargs[:name] = cargs[:name]*"_"*randstring(3)
	 new(cargs[:name],Dict(),cargs)
  end
end

"""
Plotter(type::String)
Helper function for Plotter.
"""
Plotter(type::String) = Plotter(Dict(:type => type))


"""
    fit!(plot::Plotter,features::DataFrame,labels::Vector=[])
Checks and outputs an empty layout if there are no arguments or dimensions.
# Arguments
- `plot::Plotter`: custom type
- `type::String`: input
- `axes::DataFrame`: input
- `labels::Vector=[]`: 
"""
function fit!(plot::Plotter, type::String, axes::DataFrame, labels::Vector=[])
  if axes == DataFrame()  
    return Scene()
  end
  if isempty(type)
    error("No plot type is chosen")
  end
  plot.model = plot.args
end


"""
    transform!(plot::Plotter, type::String, axes::DataFrame, color::String)
.
# Arguments
- `plot::Plotter`: custom type
- `type::String`: input
- `axes::DataFrame`: input
- `color::String`: input
"""
function transform!(plot::Plotter, type::String, axes::DataFrame, color::String)
  features = deepcopy(axes) 
  if features == DataFrame()
	 return Scene()
  end
  x = features[:,1]
  y = features[:,2]
  z = features[:,3]
# if required you can add title and axis names
  if ncol(features) == 2
    if type == "scatter"
      colors = nrow(features)
      scene = scatter(x, y, color=colors)
    end
    if type == "lines"
      scene = lines(x, y, color=color)
    end
    if type == "bar"
      scene = barplot(x, y, color=color)
    end
  if ncol(features) == 3
    if type == "scatter"
      colors = nrow(features)
      scene = scatter(x, y, z, color=colors)
    end
    if type == "lines"
      scene = lines(x, y, z, color=color)
    end
    if type == "bar"
      scene = barplot(x, y, z, color=color)
    end
  return scene
end
end
