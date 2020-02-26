module Utils

using AutoMLPipeline: Machine
using Statistics
using DataFrames
using CSV

import MLBase: Kfold

using Random: randperm


export holdout, kfold, score, infer_eltype, nested_dict_to_tuples,
       nested_dict_set!, nested_dict_merge, createmachine,
       mergedict, getiris,
       skipmean,skipmedian,skipstd,
       aggregatorclskipmissing,
       find_catnum_columns


"""
    find_nominal_columns(features::DataFrame) 

Finds all nominal columns.

Nominal columns are those that do not have Real type nor
do all their elements correspond to Real.
"""

function find_catnum_columns(instances::DataFrame, maxuniqcat::Int=0)
  nominal_columns = Int[]
  real_columns = Int[]
  for column in 1:size(instances, 2)
    vdat = instances[:, column:column] # returns a 1-column dataframe
    col_eltype = infer_eltype(vdat)
    # nominal if column type is not real or only small number of unique instances 
    # otherwise, real
    if !<:(col_eltype, Real)
      push!(nominal_columns, column)
    elseif nrow(unique(vdat)) <= maxuniqcat
      push!(nominal_columns, column)
    else
      push!(real_columns, column)
    end
  end
  return (nominal_columns,real_columns)
end

"""
    holdout(n, right_prop)
    
Holdout method that partitions a collection
into two partitions.

- `n`: Size of collection to partition
- `right_prop`: Percentage of collection placed in right partition

Returns: two partitions of indices, left and right
"""
function holdout(n, right_prop)
  shuffled_indices = randperm(n)
  partition_pivot = round(Int,right_prop * n)
  right = shuffled_indices[1:partition_pivot]
  left = shuffled_indices[partition_pivot+1:end]
  return (left, right)
end

"""
    kfold(num_instances, num_partitions)

Returns k-fold partitions.

- `num_instances`: total number of instances
- `num_partitions`: number of partitions required

Returns: training set partition.
"""
function kfold(num_instances, num_partitions)
  return collect(Kfold(num_instances, num_partitions))
end

"""
    score(metric::Symbol, actual::Vector, predicted::Vector)

Score learner predictions against ground truth values.

Available metrics:
- :accuracy

- `metric`: metric to assess with
- `actual`: ground truth values
- `predicted`: predicted values

Returns: score of learner
"""
function score(metric::Symbol, actual::Vector, predicted::Vector)
  if metric == :accuracy
    mean(actual .== predicted) * 100.0
  else
    error("Metric $metric not implemented for score.")
  end
end

"""
    infer_eltype(vector::Vector)

Returns element type of vector unless it is Any.
If Any, returns the most specific type that can be
inferred from the vector elements.

- `vector`: vector to infer element type on

Returns: inferred element type
"""
function infer_eltype(vector::Vector)
  # Obtain element type of vector
  vec_eltype = eltype(vector)

  # If element type of Vector is Any and not empty,
  # and all elements are of the same type,
  # then return the vector elements' type.
  if vec_eltype == Any && !isempty(vector)
    all_elements_same_type = all(x -> typeof(x) == typeof(first(vector)), vector)
    if all_elements_same_type
      vec_eltype = typeof(first(vector))
    end
  end

  # Return inferred element type
  return vec_eltype
end

function infer_eltype(df::DataFrame)
  infer_eltype(Matrix(df))
end

function infer_eltype(mtrx::Matrix)
  # Obtain element type of matrix
  mat_eltype = eltype(mtrx)

  # If element type of Matrix is Any and not empty,
  # and all elements are of the same type,
  # then return the matrix elements' type.
  if mat_eltype == Any && !isempty(mtrx)
    all_elements_same_type = all(x -> typeof(x) == typeof(first(mtrx)), mtrx)
    if all_elements_same_type
      mat_eltype = typeof(first(mtrx))
    end
  end

  # Return inferred element type
  return mat_eltype
end

"""
    nested_dict_to_tuples(dict::Dict)

Converts nested dictionary to list of tuples

- `dict`: dictionary that can have other dictionaries as values

Returns: list where elements are ([outer-key, inner-key, ...], value)
"""
function nested_dict_to_tuples(dict::Dict)
  set = Set()
  for (entry_id, entry_val) in dict
    if typeof(entry_val) <: Dict
      inner_set = nested_dict_to_tuples(entry_val)
      for (inner_entry_id, inner_entry_val) in inner_set
        new_entry = (vcat([entry_id], inner_entry_id), inner_entry_val)
        push!(set, new_entry)
      end
    else
      new_entry = ([entry_id], entry_val)
      push!(set, new_entry)
    end
  end
  return set
end

"""
    nested_dict_set!(dict::Dict, keys::Array{T, 1}, value) where {T}

Set value in a nested dictionary.

- `dict`: nested dictionary to assign value
- `keys`: keys to access nested dictionaries in sequence
- `value`: value to assign
"""
function nested_dict_set!(dict::Dict, keys::Array{T, 1}, value) where {T}
  inner_dict = dict
  for key in keys[1:end-1]
    inner_dict = inner_dict[key]
  end
  inner_dict[keys[end]] = value
end

"""
    nested_dict_merge(first::Dict, second::Dict)
    
Second nested dictionary is merged into first.

If a second dictionary's value as well as the first
are both dictionaries, then a merge is conducted between
the two inner dictionaries.
Otherwise the second's value overrides the first.

- `first`: first nested dictionary
- `second`: second nested dictionary

Returns: merged nested dictionary
"""
function nested_dict_merge(first::Dict, second::Dict)
  target = copy(first)
  for (second_key, second_value) in second
    values_both_dict =
      typeof(second_value) <: Dict &&
      typeof(get(target, second_key, nothing)) <: Dict
    if values_both_dict
      target[second_key] = nested_dict_merge(target[second_key], second_value)
    else
      target[second_key] = second_value
    end
  end
  return target
end

"""
    createmachine(prototype::Machine, options=nothing)

Create machine

- `prototype`: prototype machine to base new machine on
- `options`: additional options to override prototype's options

Returns: new machine
"""
function createmachine(prototype::Machine, args::Dict=Dict())
  new_args = copy(prototype.args)
  if args != Dict()
    new_args = nested_dict_merge(new_args, args)
  end

  prototype_type = typeof(prototype)
  return prototype_type(new_args)
end

"""
    aggregatorclskipmissing(fn::Function)
    
Function to create aggregator closure with skipmissing features
"""
function aggregatorclskipmissing(fn::Function)
  function skipagg(x::Union{AbstractArray,DataFrame})
    if length(collect(skipmissing(x))) == 0
      return missing
    else
      return fn(skipmissing(x))
    end
  end
  return skipagg
end


function skipmean(x::T) where {T<:Union{AbstractArray,DataFrame}} 
  if length(collect(skipmissing(x))) == 0
    missing
  else
    mean(skipmissing(x))
  end
end

function skipmedian(x::T) where {T<:Union{AbstractArray,DataFrame}} 
  if length(collect(skipmissing(x))) == 0
    missing
  else
    median(skipmissing(x))
  end
end

function skipstd(x::T) where {T<:Union{AbstractArray,DataFrame}} 
  if length(collect(skipmissing(x))) == 0
    missing
  else
    std(skipmissing(x))
  end
end


"""
    mergedict(first::Dict, second::Dict)
    
Second nested dictionary is merged into first.

If a second dictionary's value as well as the first
are both dictionaries, then a merge is conducted between
the two inner dictionaries.
Otherwise the second's value overrides the first.

- `first`: first nested dictionary
- `second`: second nested dictionary

Returns: merged nested dictionary
"""
function mergedict(first::Dict, second::Dict)
  target = copy(first)
  for (second_key, second_value) in second
    values_both_dict =
      typeof(second_value) <: Dict &&
      typeof(get(target, second_key, nothing)) <: Dict
    if values_both_dict
      target[second_key] = mergedict(target[second_key], second_value)
    else
      target[second_key] = second_value
    end
  end
  return target
end

function getiris()
  iris = CSV.read(joinpath(Base.@__DIR__,"../data","iris.csv"))
  return iris
end


end
