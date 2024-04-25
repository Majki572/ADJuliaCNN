module DenseModule

export DenseLayer, init_dense_layer, backward_pass, relu, relu_grad, identity, identity_grad

mutable struct DenseLayer
    weights::Array{Float32,2}
    biases::Array{Float32,1}
    activation::Function
    activation_grad::Function
    activations::Array{Float32,2}  # Mutable fields for runtime state
    inputs::Array{Float32,2}
end

function relu(x)
    return max.(0, x)
end

function relu_grad(x)
    return float.(x .> 0)  # Convert boolean values to 1.0 or 0.0
end

function identity(x)
    return x
end

function identity_grad(x)
    return ones(size(x))
end

function (layer::DenseLayer)(input::Array{Float32,2})
    # Store intermediate values needed for backpropagation
    layer.activations = layer.activation(layer.weights * input .+ layer.biases)
    layer.inputs = input  # Save input to use in the backward pass
    return layer.activations
end

function init_dense_layer(input_dim::Int, output_dim::Int, activation::Function, activation_grad::Function)
    weights = 0.01f0 * randn(Float32, output_dim, input_dim)
    biases = zeros(Float32, output_dim)
    # Initialize activations and inputs arrays
    activations = zeros(Float32, output_dim, 1)  # Adjust the shape appropriately
    inputs = zeros(Float32, input_dim, 1)  # Adjust the shape appropriately
    return DenseLayer(weights, biases, activation, activation_grad, activations, inputs)
end

# Implementing the backward pass for the dense layer
function backward_pass(layer::DenseLayer, d_output::Array{Float32,2})
    # Apply the derivative of the activation function to the output gradient
    d_activation = layer.activation_grad(layer.activations) .* d_output

    # Gradient w.r.t. weights
    d_weights = d_activation * layer.inputs'

    # Gradient w.r.t. biases
    d_biases = sum(d_activation, dims=2)

    # Gradient w.r.t. input
    d_input = layer.weights' * d_activation

    # Optionally, update weights and biases here or return gradients for batch updates
    return d_weights, d_biases, d_input
end

end
