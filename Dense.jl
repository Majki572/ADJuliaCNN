# Define the Dense layer function
function dense_layer_m(input::Array{Float64}, weights::Array{Float64}, biases::Array{Float64}, activation_function)
    z = weights * input .+ biases
    return activation_function.(z)  # Apply activation function element-wise
end

# Activation functions
relu(x) = max(0, x)
identity(x) = x