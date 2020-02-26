# Decision trees as found in DecisionTree Julia package.
module DecisionTreeLearners

import DecisionTree
DT = DecisionTree

# standard included modules
using DataFrames
using Random
using AutoMLPipeline.AbsTypes
using AutoMLPipeline.Utils

import AutoMLPipeline.AbsTypes: fit!, transform!
export fit!, transform!

export PrunedTree, RandomForest, Adaboost

# Pruned CART decision tree.

"""
    PrunedTree(
      Dict(
        :purity_threshold => 1.0,
        :max_depth => -1,
        :min_samples_leaf => 1,
        :min_samples_split => 2,
        :min_purity_increase => 0.0
      )
    )

Decision tree classifier.  
See [DecisionTree.jl's documentation](https://github.com/bensadeghi/DecisionTree.jl)

Hyperparmeters:
- `:purity_threshold` => 1.0 (merge leaves having >=thresh combined purity)
- `:max_depth` => -1 (maximum depth of the decision tree)
- `:min_samples_leaf` => 1 (the minimum number of samples each leaf needs to have)
- `:min_samples_split` => 2 (the minimum number of samples in needed for a split)
- `:min_purity_increase` => 0.0 (minimum purity needed for a split)

Implements `fit!`, `transform!`
"""
mutable struct PrunedTree <: Learner
  name::String
  model::Dict
  args::Dict

  function PrunedTree(args=Dict())
    default_args = Dict(
      :name => "prunetree",
      # Output to train against
      # (:class).
      :output => :class,
      # Options specific to this implementation.
      :impl_args => Dict(
        # Merge leaves having >= purity_threshold CombineMLd purity.
        :purity_threshold => 1.0,
        # Maximum depth of the decision tree (default: no maximum).
        :max_depth => -1,
        # Minimum number of samples each leaf needs to have.
        :min_samples_leaf => 1,
        # Minimum number of samples in needed for a split.
        :min_samples_split => 2,
        # Minimum purity needed for a split.
        :min_purity_increase => 0.0
      )
    )
    cargs = nested_dict_merge(default_args, args)
    cargs[:name] = cargs[:name]*"_"*randstring(3)
    new(cargs[:name],Dict(),cargs)
  end
end

"""
    fit!(tree::PrunedTree, features::DataFrame, labels::Vector) 

Optimize the hyperparameters of `PrunedTree` instance.
"""
function fit!(ptree::PrunedTree, features::DataFrame, labels::Vector) 
  instances=convert(Matrix,features)
  args = ptree.args[:impl_args]
  btreemodel = DT.build_tree(
    labels,
    instances,
    0, # num_subfeatures (keep all)
    args[:max_depth],
    args[:min_samples_leaf],
    args[:min_samples_split],
    args[:min_purity_increase])
  btreemodel = DT.prune_tree(btreemodel, args[:purity_threshold])
  ptree.model = Dict(
                    :dtmodel => btreemodel,
                    :impl_args => args
                   )
end


"""
    transform!(ptree::PrunedTree, features::DataFrame)

Predict using the optimized hyperparameters of the trained `PrunedTree` instance.
"""
function transform!(ptree::PrunedTree, features::DataFrame)
  instances=convert(Matrix,features)
  model = ptree.model[:dtmodel]
  return DT.apply_tree(model, instances)
end


# Random forest (CART).

"""
    RandomForest(
      Dict(
        :output => :class,
        :num_subfeatures => 0,
        :num_trees => 10,
        :partial_sampling => 0.7,
        :max_depth => -1
      )
    )

Random forest classification. 
See [DecisionTree.jl's documentation](https://github.com/bensadeghi/DecisionTree.jl)

Hyperparmeters:
- `:num_subfeatures` => 0  (number of features to consider at random per split)
- `:num_trees` => 10 (number of trees to train)
- `:partial_sampling` => 0.7 (fraction of samples to train each tree on)
- `:max_depth` => -1 (maximum depth of the decision trees)
- `:min_samples_leaf` => 1 (the minimum number of samples each leaf needs to have)
- `:min_samples_split` => 2 (the minimum number of samples in needed for a split)
- `:min_purity_increase` => 0.0 (minimum purity needed for a split)

Implements `fit!`, `transform!`
"""
mutable struct RandomForest <: Learner
  name::String
  model::Dict
  args::Dict
  function RandomForest(args=Dict())
    default_args = Dict(
      :name => "rf",
      # Output to train against
      # (:class).
      :output => :class,
      # Options specific to this implementation.
      :impl_args => Dict(
        # Number of features to train on with trees (default: 0, keep all).
        :num_subfeatures => 0,
        # Number of trees in forest.
        :num_trees => 10,
        # Proportion of trainingset to be used for trees.
        :partial_sampling => 0.7,
        # Maximum depth of each decision tree (default: no maximum).
        :max_depth => -1
      )
    )
    cargs = nested_dict_merge(default_args, args)
    cargs[:name] = cargs[:name]*"_"*randstring(3)
    new(cargs[:name],Dict(),cargs)
  end
end


"""
    fit!(forest::RandomForest, features::T, labels::Vector) where {T<:Union{Vector,Matrix,DataFrame}}

Optimize the parameters of the `RandomForest` instance.
"""
function fit!(forest::RandomForest, features::DataFrame, labels::Vector) 
  instances=convert(Matrix,features)
  # Set training-dependent options
  impl_args = forest.args[:impl_args]
  # Build model
  model = DT.build_forest(
    labels, 
    instances,
    impl_args[:num_subfeatures],
    impl_args[:num_trees],
    impl_args[:partial_sampling],
    impl_args[:max_depth]
  )
  forest.model = Dict(
                      :dtmodel => model,
                      :impl_args => impl_args
                     )
end


"""
    transform!(forest::RandomForest, features::T) where {T<:Union{Vector,Matrix,DataFrame}}


Predict using the optimized hyperparameters of the trained `RandomForest` instance.
"""
function transform!(forest::RandomForest, features::DataFrame)
  instances = features
  instances=convert(Matrix,features)
  model = forest.model[:dtmodel]
  return DT.apply_forest(model, instances)
end


# Adaboosted decision stumps.

"""
    Adaboost(
      Dict(
        :output => :class,
        :num_iterations => 7
      )
    )

Adaboosted decision tree stumps. See
[DecisionTree.jl's documentation](https://github.com/bensadeghi/DecisionTree.jl)

Hyperparameters:
- `:num_iterations` => 7 (number of iterations of AdaBoost)

Implements `fit!`, `transform!`
"""
mutable struct Adaboost <: Learner
  name::String
  model::Dict
  args::Dict
  function Adaboost(args=Dict())
    default_args = Dict(
      :name => "adaboost",
      # Output to train against
      # (:class).
      :output => :class,
      # Options specific to this implementation.
      :impl_args => Dict(
        # Number of boosting iterations.
        :num_iterations => 7
      )
    )
    cargs = nested_dict_merge(default_args, args)
    cargs[:name] = cargs[:name]*"_"*randstring(3)
    new(cargs[:name],Dict(),cargs)
  end
end


"""
    fit!(adaboost::Adaboost, features::DataFrame, labels::Vector) 

Optimize the hyperparameters of `Adaboost` instance.
"""
function fit!(adaboost::Adaboost, features::DataFrame, labels::Vector) 
  instances = convert(Matrix,features)
  # NOTE(svs14): Variable 'model' renamed to 'ensemble'.
  #              This differs to DecisionTree
  #              official documentation to avoid confusion in variable
  #              naming within CombineML.
  ensemble, coefficients = DT.build_adaboost_stumps(
    labels, instances, adaboost.args[:impl_args][:num_iterations]
  )
  adaboost.model = Dict(
    :ensemble => ensemble,
    :coefficients => coefficients
  )
end

"""
    transform!(adaboost::Adaboost, features::T) where {T<:Union{Vector,Matrix,DataFrame}}

Predict using the optimized hyperparameters of the trained `Adaboost` instance.
"""
function transform!(adaboost::Adaboost, features::DataFrame)::Vector{<:Any}
  instances = convert(Matrix,features)
  return DT.apply_adaboost_stumps(
    adaboost.model[:ensemble], adaboost.model[:coefficients], instances
  )
end


end # module
