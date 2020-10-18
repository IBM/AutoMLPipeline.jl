module BaseFilters

using Infiltrator
using Random
using Dates
using DataFrames
using Statistics

using AutoMLPipeline.Utils
using AutoMLPipeline.AbsTypes: Machine, Transformer, Learner, Workflow, Computer

import AutoMLPipeline.AbsTypes: fit!, transform!
export fit!,transform!
export OneHotEncoder, Imputer


"""
    OneHotEncoder(Dict(
       # Nominal columns
       :nominal_columns => Int[],

       # Nominal column values map. Key is column index, value is list of
       # possible values for that column.
       :nominal_column_values_map => Dict{Int,Any}()
    ))

Transforms myinstances with nominal features into one-hot form
and coerces the instance matrix to be of element type Float64.

Implements `fit!` and `transform`.
"""
mutable struct OneHotEncoder <: Transformer
  name::String
  model::Dict
  args::Dict

  function OneHotEncoder(args::Dict=Dict())
    default_args = Dict(
                        :name => "ohe",
                        # Nominal columns
                        :nominal_columns => Int[],
                        # Nominal column values map. Key is column index, value is list of
                        # possible values for that column.
                        :nominal_column_values_map => Dict{Int,Any}() 
                       )
    cargs=nested_dict_merge(default_args,args)
    cargs[:name] = cargs[:name]*"_"*randstring(3)
    new(cargs[:name],Dict(),cargs)
  end
end

function fit!(ohe::OneHotEncoder, myinstances::DataFrame, labels::Vector=[]) 
  # Obtain nominal columns
  nominal_columns = ohe.args[:nominal_columns]
  if nominal_columns == Int[]
    nominal_columns,_ = find_catnum_columns(myinstances)
  end

  # Obtain unique values for each nominal column
  nominal_column_values_map = ohe.args[:nominal_column_values_map]
  if nominal_column_values_map == Dict{Int,Any}()
    for column in nominal_columns
      nominal_column_values_map[column] = unique(myinstances[:, column])
    end
  end

  # Create model
  ohe.model = Dict(
    :nominal_columns => nominal_columns,
    :nominal_column_values_map => nominal_column_values_map
  )
end

function transform!(ohe::OneHotEncoder, pinstances::DataFrame)
  myinstances = deepcopy(pinstances)
  nominal_columns = ohe.model[:nominal_columns]
  nominal_column_values_map = ohe.model[:nominal_column_values_map]

  # Create new transformed instance matrix of type Float64
  num_rows = size(myinstances, 1)
  num_columns = (size(myinstances, 2) - length(nominal_columns))
  if !isempty(nominal_column_values_map)
    num_columns += sum(map(x -> length(x), values(nominal_column_values_map)))
  end
  transformed_instances = zeros(Float64, num_rows, num_columns)

  # Fill transformed instance matrix
  col_start_index = 1
  for column in 1:size(myinstances, 2)
    if !in(column, nominal_columns)
      transformed_instances[:, col_start_index] = myinstances[:, column]
      col_start_index += 1
    else
      col_values = nominal_column_values_map[column]
      for row in 1:size(myinstances, 1)
        entry_value = myinstances[row, column]
        entry_value_index = findfirst(isequal(entry_value),col_values)
        if entry_value_index == 0 || entry_value_index == nothing
          @warn "Unseen value found in OneHotEncoder,
                for entry ($row, $column) = $(entry_value).
                Patching value to $(col_values[1])."
          entry_value_index = 1
        end
        entry_column = (col_start_index - 1) + entry_value_index
        transformed_instances[row, entry_column] = 1
      end
      col_start_index += length(nominal_column_values_map[column])
    end
  end

  return transformed_instances |> DataFrame
end


"""
    Imputer(
       Dict(
          # Imputation strategy.
          # Statistic that takes a vector such as mean or median.
          :strategy => mean
       )
    )

Imputes NaN values from Float64 features.

Implements `fit!` and `transform`.
"""
mutable struct Imputer <: Transformer
  name::String
  model::Dict
  args::Dict

  function Imputer(args=Dict())
    default_args = Dict(
      # Imputation strategy.
      # Statistic that takes a vector such as mean or median.
      :strategy => mean
    )
    cargs=nested_dict_merge(default_args,args)
    cargs[:name] = cargs[:name]*"_"*randstring(3)
    new(cargs[:name],Dict(),cargs)
  end
end

function fit!(imp::Imputer, myinstances::DataFrame, labels::Vector=[]) 
  imp.model = imp.args
end

function transform!(imp::Imputer, myinstances::DataFrame) 
  new_instances = deepcopy(myinstances)
  strategy = imp.model[:strategy]

  for column in 1:size(myinstances, 2)
    column_values = myinstances[:, column]
    col_eltype = infer_eltype(column_values)

    if <:(col_eltype, Real)
      na_rows = map(x -> isnan(x), column_values)
      if any(na_rows)
        fill_value = strategy(column_values[.!na_rows])
        new_instances[na_rows, column] .= fill_value
      end
    end
  end
  return new_instances |> DataFrame
end

"""
    Wrapper(
       default_args = Dict(
          # Transformer to call.
          :transformer => OneHotEncoder(),
          # Transformer args.
          :transformer_args => Dict()
       )
    )
       
Wraps around a AutoMLPipeline transformer.

Implements `fit!` and `transform`.
"""
mutable struct Wrapper <: Transformer
  name::String
  model::Dict
  args::Dict

  function Wrapper(args=Dict())
    default_args = Dict(
      # Transformer to call.
      :transformer => OneHotEncoder(),
      # Transformer args.
      :transformer_args => Dict()
    )
    cargs=nested_dict_merge(default_args,args)
    cargs[:name] = cargs[:name]*"_"*randstring(3)
    new(cargs[:name],Dict(),cargs)
  end
end

function fit!(wrapper::Wrapper, myinstances::DataFrame, labels::Vector) 
  transformer_args = wrapper.args[:transformer_args]
  transformer = createtransformer(
    wrapper.args[:transformer],
    transformer_args
  )

  if transformer_args != Dict()
    transformer_args = mergedict(transformer.args, transformer_args)
  end
  fit!(transformer, myinstances, labels)

  wrapper.model = Dict(
    :transformer => transformer,
    :transformer_args => transformer_args
  )
end

function transform!(wrapper::Wrapper, myinstances::DataFrame)
  transformer = wrapper.model[:transformer]
  return transform!(transformer, myinstances) 
end

"""
    createtransformer(prototype::Transformer, args=Dict())

Create transformer

- `prototype`: prototype transformer to base new transformer on
- `options`: additional options to override prototype's options

Returns: new transformer.
"""
function createtransformer(prototype::Transformer, args=Dict())
  new_args = copy(prototype.args)
  if args != Dict()
    new_args = mergedict(new_args, args)
  end

  prototype_type = typeof(prototype)
  return prototype_type(new_args)
end


end
