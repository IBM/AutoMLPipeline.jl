module AbsTypes

using DataFrames

export fit!, transform!, fit_transform!
export Machine, Learner, Transformer, Workflow, Computer

abstract type Machine end
abstract type Computer <: Machine end # does computation: learner and transformer
abstract type Workflow <: Machine end # types: Linear vs Combine
abstract type Learner <: Computer end
abstract type Transformer <: Computer end

# multiple dispatch for fit
"""
    fit!(mc::Machine, input::DataFrame, output::Vector)

Generic trait to be overloaded by different subtypes of Machine.
"""
function fit!(mc::Machine, input::DataFrame, output::Vector)
	error(typeof(mc),"not implemented")
end


"""
    transform!(mc::Machine, input::DataFrame)

Generic trait to be overloaded by different subtypes of Machine.
"""
function transform!(mc::Machine, input::DataFrame)
	error(typeof(mc),"not implemented")
end

# dynamic dispatch based Machine subtypes
"""
    fit_transform!(mc::Machine, input::DataFrame, output::Vector=Vector())

Dynamic dispatch that calls in sequence `fit!` and `transform!` functions.
"""
function fit_transform!(mc::Machine, input::DataFrame, output::Vector=Vector())
	fit!(mc,input,output)
	transform!(mc,input)
end

end

