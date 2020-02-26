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
    if features == DataFrame()
	error("empty dataframe")
    end
    return features[:,ft.model[:columns]]
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
    catcols = ft.model[:nominal_columns]
    return features[:,catcols]
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
    realcols = ft.model[:real_columns]
    return features[:,realcols]
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
    catcols = ft.model[:nominal_columns]
    features[!,catcols] .= features[!,catcols] .|> string
    return features
end

function feature_test()
    data = getiris()
    X = data[:,1:5]
    X[!,5] = X[!,5] .|> string
    catfeat = FeatureSelector([5])
    numfeat = FeatureSelector([1,2,3,4])
    autocat = CatFeatureSelector()
    autonum = NumFeatureSelector()
    @assert (fit_transform!(catfeat,X) .== X[:,5]) |> Matrix |> sum == 150
    @assert (fit_transform!(numfeat,X) .== X[:,1:4]) |> Matrix |> sum == 600
    @assert (fit_transform!(autocat,X) .== X[:,5]) |> Matrix |> sum == 150
    @assert (fit_transform!(autonum,X) .== X[:,1:4]) |> Matrix |> sum == 600
    catnumdata = hcat(X,repeat([1,2,3,4,5],30))
    catnum = CatNumDiscriminator()
    fit_transform!(catnum,catnumdata)
    @assert eltype(catnumdata[:,6]) <: String
    catnumdata = hcat(X,repeat([1,2,3,4,5],30))
    catnum = CatNumDiscriminator(0)
    fit_transform!(catnum,catnumdata)
    @assert eltype(catnumdata[:,6]) <: Int
end

end
