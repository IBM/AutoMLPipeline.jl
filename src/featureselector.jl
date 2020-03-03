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
    return nfeatures[:,ft.model[:columns]]
end

# ----------
# automatically extracts cat features based on their inferred element non-number types
mutable struct CatFeatureSelector <: Transformer
    name::String
    model::Dict
    args::Dict

    function CatFeatureSelector(args::Dict = Dict())
	default_args = Dict(
			    :name => "catfeatsel",
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
    return nfeatures[:,catcols]
end

# ---------
# automatically extracts numeric features based on their inferred element types
mutable struct NumFeatureSelector <: Transformer
    name::String
    model::Dict
    args::Dict

    function NumFeatureSelector(args::Dict = Dict())
	default_args = Dict(
			    :name => "numfeatsel"
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
    return nfeatures[:,realcols]
end


# ---------
# convert numeric categories to string based on count of unique elements
mutable struct CatNumDiscriminator <: Transformer
    name::String
    model::Dict
    args::Dict

    function CatNumDiscriminator(args::Dict = Dict())
	default_args = Dict(
			    :name => "catnumdisc",
			    # default max categories for numeric-encoded categories
			    :maxcategories => 24,
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
    nfeatures[!,catcols] .= nfeatures[!,catcols] .|> string
    return nfeatures
end

end
