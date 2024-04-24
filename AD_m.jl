# Define a basic structure for storing variable information including gradients
mutable struct Var
    value::Float64
    grad::Float64
    func::Function
    args::Vector{Ref{Var}}

    Var(value; grad=0.0, func=identity, args=[]) = new(value, grad, func, args)
end

# Define how variables interact through operations
Base.:*(a::Var, b::Var) = Var(a.value * b.value, func=*, args=[Ref(a), Ref(b)])
Base.:+(a::Var, b::Var) = Var(a.value + b.value, func=+, args=[Ref(a), Ref(b)])

# Define the backward pass to propagate gradients
function backward!(y::Var)
    y.grad = 1.0  # seed gradient
    stack = [y]
    while !isempty(stack)
        current_var = pop!(stack)
        for arg_ref in current_var.args
            arg = arg_ref[]
            if current_var.func == *
                arg.grad += current_var.grad * (arg === arg_ref[] ? current_var.args[2][] : current_var.args[1][]).value
            elseif current_var.func == +
                arg.grad += current_var.grad
            end
            push!(stack, arg)
        end
    end
end
