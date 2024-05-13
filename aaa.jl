module DenseModule
using Random

export DenseLayer, init_dense_layer, backward_pass, relu, relu_grad, identity, identity_grad

mutable struct DenseLayer
    weights::Array{Float32,2}
    biases::Array{Float32,1}
    grad_weights::Union{Nothing,Array{Float32,2}}
    grad_biases::Union{Nothing,Array{Float32,1}}
    activation::Function
    activation_grad::Function
    activations::Array{Float32,2}
    inputs::Array{Float32,2}
end

function relu(x)
    for i in eachindex(x)
        x[i] = max(0, x[i])
    end
    return x
end

function relu_grad(x)
    iterator = eachindex(x)
    for i in iterator
        x[i] = Float32(x[i] > 0)
    end
    return x
end

function identity(x)
    return x
end

function identity_grad(x)
    return ones(size(x))
end

function (layer::DenseLayer)(input::Array{Float32,2})
    z = layer.weights * input
    z_sizeone = size(z, 1)
    z_sizetwo = size(z, 2)

    for i in 1:z_sizeone
        for j in 1:z_sizetwo
            layer.activations[i, j] = z[i, j] + layer.biases[i]
        end
    end

    # Store intermediate values needed for backpropagation
    layer.activations = layer.activation(layer.activations)
    layer.inputs = input  # Save input to use in the backward pass
    return layer.activations
end

function init_dense_layer(input_dim::Int, output_dim::Int, activation::Function, activation_grad::Function, seedy::Int)

    # seed = rand(UInt32)
    Random.seed!(seedy)

    weights = 0.178 * randn(Float32, output_dim, input_dim)
    biases = zeros(Float32, output_dim)
    grad_weights = zeros(Float32, output_dim, input_dim)
    grad_biases = zeros(Float32, output_dim)
    activations = zeros(Float32, output_dim, 1)
    inputs = zeros(Float32, input_dim, 1)
    return DenseLayer(weights, biases, grad_weights, grad_biases, activation, activation_grad, activations, inputs)
end

function backward_pass(layer::DenseLayer, d_output::Array{Float32,2})

    a_grad = layer.activation_grad(layer.activations)
    a_grad_sizeone = size(a_grad, 1)
    a_grad_sizetwo = size(a_grad, 2)
    d_activation = zeros(Float32, a_grad_sizeone, a_grad_sizetwo)
    # Apply the derivative of the activation function
    for i in 1:a_grad_sizeone
        for j in 1:a_grad_sizetwo
            d_activation[i, j] = a_grad[i, j] * d_output[i, j]
        end
    end #d_activation = layer.activation_grad(layer.activations) .* d_output

    # Calculate gradients
    d_weights = d_activation * layer.inputs'
    d_biases = sum(d_activation, dims=2)
    d_input = layer.weights' * d_activation

    a = size(layer.grad_weights, 1)
    b = size(layer.grad_weights, 2)
    for i in 1:a
        for j in 1:b
            layer.grad_weights[i, j] += d_weights[i, j]
        end
    end
    length_biases = length(layer.grad_biases)
    for i in 1:length_biases
        layer.grad_biases[i] += d_biases[i]
    end

    return d_input
end

end
