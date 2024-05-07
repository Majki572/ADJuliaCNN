module ConvolutionModule

mutable struct ConvLayer
    weights::Array{Float32,4}
    biases::Array{Float32,1}
    grad_weights::Array{Float32,4}
    grad_biases::Array{Float32,1}
    stride::Int
    padding::Int
    last_input::Union{Nothing,Array{Float32,3}}  # Adding this field to store the input
end

function forward(input::Array{Float32,3}, kernels::Array{Float32,4}, stride::Int, padding::Int)
    # Input dimensions
    (height, width, channels) = size(input)

    # Kernel dimensions
    (kernel_height, kernel_width, _, num_kernels) = size(kernels)

    # Output dimensions
    out_height = div(height - kernel_height + 2 * padding, stride) + 1
    out_width = div(width - kernel_width + 2 * padding, stride) + 1

    # Apply padding
    padded_input = zeros(Float32, height + 2 * padding, width + 2 * padding, channels)
    padded_input[padding+1:padding+height, padding+1:padding+width, :] = input

    # Output initialization
    output = zeros(Float32, out_height, out_width, num_kernels)

    # Perform convolution for each filter
    for k in 1:num_kernels
        kernel = kernels[:, :, :, k]
        for h in 1:stride:height-kernel_height+1+2*padding
            for w in 1:stride:width-kernel_width+1+2*padding
                patch = padded_input[h:h+kernel_height-1, w:w+kernel_width-1, :]
                output[div(h - 1, stride)+1, div(w - 1, stride)+1, k] += sum(patch .* kernel)
            end
        end
    end

    return output
end

function (cl::ConvLayer)(input::Array{Float32,3})
    cl.last_input = copy(input)  # Store the original input for use in the backward pass

    # Perform convolution
    conv_output = forward(input, cl.weights, cl.stride, cl.padding)

    # Add bias (broadcasting addition across channels)
    for c in axes(conv_output, 3)
        conv_output[:, :, c] .+= cl.biases[c]
    end

    # Apply ReLU activation function
    return relu(conv_output)
end

# Implementing ReLU activation function
function relu(x)
    return max.(0, x)
end

function init_conv_layer(kernel_height::Int, kernel_width::Int, input_channels::Int, output_channels::Int, stride::Int, padding::Int)
    weights = randn(Float32, kernel_height, kernel_width, input_channels, output_channels)
    biases = zeros(Float32, output_channels)
    grad_weights = zeros(Float32, kernel_height, kernel_width, input_channels, output_channels)
    grad_biases = zeros(Float32, output_channels)
    return ConvLayer(weights, biases, grad_weights, grad_biases, stride, padding, nothing)
end

function backward_pass(cl::ConvLayer, grad_output::Array{Float32,3})
    input = cl.last_input
    (height, width, channels) = size(input)
    (kernel_height, kernel_width, _, num_kernels) = size(cl.weights)
    stride, padding = cl.stride, cl.padding

    grad_input = zeros(Float32, size(input))
    # Prepare padded input and gradients for input
    padded_input = zeros(Float32, height + 2 * padding, width + 2 * padding, channels)
    padded_input[padding+1:end-padding, padding+1:end-padding, :] = input

    for k in 1:num_kernels
        for h in 1:stride:height-kernel_height+1+2*padding
            for w in 1:stride:width-kernel_width+1+2*padding
                h_out = div(h - 1, stride) + 1
                w_out = div(w - 1, stride) + 1
                if h_out <= size(grad_output, 1) && w_out <= size(grad_output, 2)
                    patch = padded_input[h:h+kernel_height-1, w:w+kernel_width-1, :]
                    grad_bias = grad_output[h_out, w_out, k]
                    cl.grad_biases[k] += grad_bias
                    cl.grad_weights[:, :, :, k] += patch * grad_bias
                end
            end
        end
    end

    # Update weights and biases here
    # cl.weights .-= 0.01 * grad_weights
    # cl.biases .-= 0.01 * grad_biases
    return Float32.(grad_input) #, grad_weights, grad_biases
end


end
