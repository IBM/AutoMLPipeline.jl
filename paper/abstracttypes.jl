abstract type Computer    <: Machine  
abstract type Workflow    <: Machine  
abstract type Learner     <: Computer  
abstract type Transformer <: Computer  

function fit!(mc::Machine, input::DataFrame, output::Vector)
   error(typeof(mc)," has no implementation.")
end

function transform!(mc::Machine, input::DataFrame)
   error(typeof(mc)," has no implementation.")
end

function fit_transform!(mc::Machine, input::DataFrame, output::Vector)
   fit!(mc,input,output)
   transform!(mc,input)
end


