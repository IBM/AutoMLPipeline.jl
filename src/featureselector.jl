module FeatureSelectors

using DataFrames
using Random

using ..AbsTypes
using ..BaseFilters
using ..Utils

import ..AbsTypes: fit!, transform!
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
   model::Dict{Symbol,Any}

   function FeatureSelector(args::Dict = Dict{Symbol,Any}())
      default_args = Dict{Symbol,Any}(
         :name => "featureselector",
         :columns => Int[]
      )
      cargs=nested_dict_merge(default_args,args)
      cargs[:name] = cargs[:name]*"_"*randstring(3)
      new(cargs[:name],cargs)
   end
end

"""
    FeatureSelector(cols::Vector{Int})

Helper function for FeatureSelector.
"""
function FeatureSelector(cols::Vector{Int}) 
   FeatureSelector(Dict(:columns => cols))
end

"""
    FeatureSelector(cols::Vararg{Int})

Helper function for FeatureSelector.
"""
FeatureSelector(cols::Vararg{Int}) = FeatureSelector([cols...])

function fit!(ft::FeatureSelector, features::DataFrame, labels::Vector=[])
   if features == DataFrame()
      error("empty dataframe")
   end
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
    CatFeatureSelector(Dict(:name => "catf"))

Automatically extract categorical columns based on 
inferred element types.

Implements `fit!` and `transform!`.
"""
mutable struct CatFeatureSelector <: Transformer
   name::String
   model::Dict{Symbol,Any}

   function CatFeatureSelector(args::Dict = Dict{Symbol,Any}())
      default_args = Dict{Symbol,Any}(
         :name => "catf",
         :nominal_columns => Int[]
      )
      cargs=nested_dict_merge(default_args,args)
      cargs[:name] = cargs[:name]*"_"*randstring(3)
      new(cargs[:name],cargs)
   end
end

function fit!(ft::CatFeatureSelector, features::DataFrame, labels::Vector=[])
    if features == DataFrame()
        error("empty dataframe")
    end
    catcols,_ = find_catnum_columns(features)

    # create model
    ft.model[:nominal_columns] = catcols
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
   model::Dict{Symbol,Any}

   function NumFeatureSelector(args::Dict = Dict())
      default_args = Dict{Symbol,Any}(
        :name => "numf",
        :numcols => Int[]
      )
      cargs=nested_dict_merge(default_args,args)
      cargs[:name] = cargs[:name]*"_"*randstring(3)
      new(cargs[:name],cargs)
   end
end

function fit!(ft::NumFeatureSelector, features::DataFrame, labels::Vector=[])
    if features == DataFrame()
        error("empty dataframe")
    end
    _,numcols = find_catnum_columns(features)

    # create model
    ft.model[:numcols] = numcols
end

function transform!(ft::NumFeatureSelector, features::DataFrame)
   nfeatures = deepcopy(features)
   numcols = ft.model[:numcols]
   if numcols != [] 
      return nfeatures[:,numcols]
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

   function CatNumDiscriminator(args::Dict = Dict())
      default_args = Dict(
         :name => "catnumdisc",
         # default max categories for numeric-encoded categories
         :maxcategories => 24,
         :nominal_columns => Int[],
         :numcols => Int[]
      )
      cargs=nested_dict_merge(default_args,args)
      cargs[:name] = cargs[:name]*"_"*randstring(3)
      new(cargs[:name],cargs)
   end
end

"""
    CatNumDiscriminator(maxcat::Int)

Helper function for CatNumDiscriminator.
"""
function CatNumDiscriminator(maxcat::Int)
    CatNumDiscriminator(Dict(:maxcategories=>maxcat))
end

function fit!(ft::CatNumDiscriminator, features::DataFrame, labels::Vector=[])
    if features == DataFrame()
        error("empty dataframe")
    end
    catcols,numcols = find_catnum_columns(features,ft.model[:maxcategories])

    # create model
    ft.model[:numcols] = numcols
    ft.model[:nominal_columns] = catcols
end

function transform!(ft::CatNumDiscriminator, features::DataFrame)
    nfeatures = features |> deepcopy
    catcols = ft.model[:nominal_columns]
    if catcols != [] 
		nfeatures[!,catcols] = nfeatures[!,catcols] .|> string
    end
    return nfeatures
end

end
