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
        :dimension => dim,
        :axes => [x, y, ...]
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
Plotter(dimension::Int)
Helper function for Plotter.
"""
Plotter(dimension::Int) = Plotter(Dict(:dim => dimension))


"""
    fit!(plot::Plotter,features::DataFrame,labels::Vector=[])
Checks and outputs an empty layout if there are no arguments or dimensions.
# Arguments
- `plot::Plotter`: custom type
- `dim::Int`: input
- `features::DataFrame`: input
- `labels::Vector=[]`: 
"""
function fit!(plot::Plotter, features::DataFrame, labels::Vector=[])
  if xfeature == DataFrame() || yfeature == DataFrame()
         return Scene()
  end
  plot.model = plot.args
end


"""
    transform!(plot::Plotter, dim::Int, nfeatures::DataFrame, color::String)
.
# Arguments
- `plot::Plotter`: custom type
- `dim::Int`: input
- `nfeatures::DataFrame`: input
- `color::String`: input
"""
function transform!(plot::Plotter, dim::Int, nfeatures::DataFrame, color::String)
  features = deepcopy(nfeatures) 
  if features == DataFrame()
	 return Scene()
  end
  r = nrow(features)
  x = features[:,1]
  y = features[:,2]
  z = features[:,3]
  if dim == 2
     scene = Scene(x, y, color:color)
  end
  if dim == 3
     scene = Scene(x, y, z, color:color)
  end
  return scene
end

end
