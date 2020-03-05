module FeatureSelectors

using DataFrames
using Random

using AutoMLPipeline.AbsTypes
using AutoMLPipeline.BaseFilters
using AutoMLPipeline.Utils

import AutoMLPipeline.AbsTypes: fit!, transform!
export fit!, transform!
export FeatureSelector, CatFeatureSelector, NumFeatureSelector, CatNumDiscriminator

export feature_test

# generic way to extract num/cat features by specifying their columns
"""
    FeatureSelector(
       Dict(
         :name => "featureselector",
	 :columns => [col1, col2, ...]
       )
    )

Returns a dataframe of the selected columns.

Implements `fit!` and `transform!`.
"""
mutable struct FeatureSelector <: Transformer
    name::String
    model::Dict
    args::Dict

    function FeatureSelector(args::Dict = Dict())
	default_args = Dict(
			    :name => "featureselector",
			    :columns => Int[],
			    )
	cargs=nested_dict_merge(default_args,args)
	cargs[:name] = cargs[:name]*"_"*randstring(3)
	new(cargs[:name],Dict(),cargs)
    end
end

function FeatureSelector(cols::Vector{Int})
    FeatureSelector(Dict(:columns => cols))
end

function fit!(ft::FeatureSelector, features::DataFrame, labels::Vector=[])
    if features == DataFrame()
	error("empty dataframe")
    end
    ft.model = ft.args
end

function transform!(ft::FeatureSelector, features::DataFrame)
    nfeatures = deepcopy(features) 
    if nfeatures == DataFrame()
	error("empty dataframe")
    end
    cols = ft.model[:columns]
    if  cols != []
	return nfeatures[:,cols]
    else
	return DataFrame()
    end
end

# ----------
"""
    CatFeatureSelector(Dict(:name => "catfeatsel"))

Automatically extract categorical columns based on 
inferred element types.

Implements `fit!` and `transform!`.
"""
mutable struct CatFeatureSelector <: Transformer
    name::String
    model::Dict
    args::Dict

    function CatFeatureSelector(args::Dict = Dict())
	default_args = Dict(
			    :name => "catfeatsel",
			    :nominal_columns => []
		       )
	cargs=nested_dict_merge(default_args,args)
	cargs[:name] = cargs[:name]*"_"*randstring(3)
	new(cargs[:name],Dict(),cargs)
    end
end

function fit!(ft::CatFeatureSelector, features::DataFrame, labels::Vector=[])
    if features == DataFrame()
        error("empty dataframe")
    end
    catcols,_ = find_catnum_columns(features)

    # create model
    ft.model = Dict(
		    :nominal_columns => catcols
	       )
end

function transform!(ft::CatFeatureSelector, features::DataFrame)
    nfeatures = deepcopy(features)
    catcols = ft.model[:nominal_columns]
    if catcols != []
	return nfeatures[:,catcols]
    else
	return DataFrame()
    end
end

"""
    NumFeatureSelector(Dict(:name=>"numfeatsel"))

Automatically extracts numeric features based on their inferred element types.

Implements `fit!` and `transform!`.
"""
mutable struct NumFeatureSelector <: Transformer
    name::String
    model::Dict
    args::Dict

    function NumFeatureSelector(args::Dict = Dict())
	default_args = Dict(
			    :name => "numfeatsel",
			    :real_columns => []
			    )
	cargs=nested_dict_merge(default_args,args)
	cargs[:name] = cargs[:name]*"_"*randstring(3)
	new(cargs[:name],Dict(),cargs)
    end
end

function fit!(ft::NumFeatureSelector, features::DataFrame, labels::Vector=[])
    if features == DataFrame()
        error("empty dataframe")
    end
    _,realcols = find_catnum_columns(features)

    # create model
    ft.model = Dict(
		    :real_columns => realcols
		    )
end

function transform!(ft::NumFeatureSelector, features::DataFrame)
    nfeatures = deepcopy(features)
    realcols = ft.model[:real_columns]
    if realcols != [] 
	return nfeatures[:,realcols]
    else
	return DataFrame()
    end
end

"""
    CatNumDiscriminator(
       Dict(
          :name => "catnumdisc",
          :maxcategories => 24
       )
    )

Transform numeric columns to string (as categories) 
if the count of their unique elements <= maxcategories.

Implements `fit!` and `transform!`.
"""
mutable struct CatNumDiscriminator <: Transformer
    name::String
    model::Dict
    args::Dict

    function CatNumDiscriminator(args::Dict = Dict())
	default_args = Dict(
			    :name => "catnumdisc",
			    # default max categories for numeric-encoded categories
			    :maxcategories => 24,
			    :nominal_columns => Int[],
			    :real_columns => Int[]
		       )
	cargs=nested_dict_merge(default_args,args)
	cargs[:name] = cargs[:name]*"_"*randstring(3)
	new(cargs[:name],Dict(),cargs)
    end
end

function CatNumDiscriminator(maxcat::Int)
    CatNumDiscriminator(Dict(:maxcategories=>maxcat))
end

function fit!(ft::CatNumDiscriminator, features::DataFrame, labels::Vector=[])
    if features == DataFrame()
        error("empty dataframe")
    end
    catcols,realcols = find_catnum_columns(features,ft.args[:maxcategories])

    # create model
    ft.model = Dict(
	:real_columns => realcols,
	:nominal_columns => catcols
    )
end

function transform!(ft::CatNumDiscriminator, features::DataFrame)
    nfeatures = features |> deepcopy
    catcols = ft.model[:nominal_columns]
    if catcols != [] 
	nfeatures[!,catcols] .= nfeatures[!,catcols] .|> string
    end
    return nfeatures
end

end
