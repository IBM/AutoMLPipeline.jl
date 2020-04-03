module NARemovers

using Random
using DataFrames

using AutoMLPipeline.AbsTypes
using AutoMLPipeline.BaseFilters
using AutoMLPipeline.Utils

import AutoMLPipeline.AbsTypes: fit!, transform!
export fit!, transform!
export NARemover


"""
    NARemover(
       Dict(
         :name => "nadetect",
         :acceptance => 0.10 # tolerable NAs percentage
       )
    )

Removes columns with NAs greater than acceptance rate.
This assumes that it processes columns of features. 
The output column should not be part of input to avoid
it being excluded if it fails the acceptance critera.

Implements `fit!` and `transform!`.
"""
mutable struct NARemover <: Transformer
    name::String
    model::Dict
    args::Dict

    function NARemover(args::Dict = Dict())
	default_args = Dict(
			    :name => "nadetect",
			    :acceptance => 0.10
			    )
	cargs=nested_dict_merge(default_args,args)
	cargs[:name] = cargs[:name]*"_"*randstring(3)
	new(cargs[:name],Dict(),cargs)
    end
end

"""
    NARemover(acceptance::Float64)

Helper function for NARemover.
"""
NARemover(acceptance::Float64) = NARemover(Dict(:acceptance => acceptance))

function fit!(nad::NARemover, features::DataFrame, labels::Vector=[])
    if features == DataFrame()
	error("empty dataframe")
    end
    nad.model = nad.args
end


function transform!(nad::NARemover, nfeatures::DataFrame)
    features = deepcopy(nfeatures) 
    if features == DataFrame()
	error("empty dataframe")
    end
    sz = nrow(features)
    tol = nad.model[:acceptance]
    colnames = []
    for (colname,dat) in eachcol(features,true)
	if sum(ismissing.(dat)) < tol*sz
	    push!(colnames,colname)
	end
    end
    xtr =  features[:,colnames]
    return xtr
end

end

