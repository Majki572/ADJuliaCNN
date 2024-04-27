struct CompNode
    value::Array{Float64,3}  # The computed value for this node
    grad::Array{Float64,3}   # Gradient of the loss with respect to this node
    inputs::Vector{CompNode}  # Input nodes to this node
    backward::Function        # Function to call on backward pass
end


function forward(node::GradNode)
    node.value = node.func(map(x -> x.value, node.inputs))
    return node.value
end

function backward(node::GradNode, grad::Float64=1.0)
    node.grad += grad
    for (input, partial_grad_func) in zip(node.inputs, node.func′(node.inputs))
        backward(input, grad * partial_grad_func(input.value))
    end
end

# Example usage
x = GradNode(2.0, 0.0, identity, [])
y = GradNode(3.0, 0.0, identity, [])
z = GradNode(0.0, 0.0, (inputs) -> inputs[1] * inputs[2], [x, y])

forward(z)
println("z: ", z.value)  # Output of multiplication
backward(z)
println("dz/dx: ", x.grad)  # Gradient of z with respect to x
println("dz/dy: ", y.grad)  # Gradient of z with respect to y
