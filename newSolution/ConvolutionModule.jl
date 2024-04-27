module ConvolutionModule

mutable struct ConvLayer
    weights::Array{Float32,4}
    biases::Array{Float32,1}
    stride::Int
    padding::Int
    last_input::Union{Nothing,Array{Float32,3}}  # Adding this field to store the input
end

function conv2d(input::Array{Float32,3}, kernels::Array{Float32,4}, stride::Int, padding::Int)
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

# Implementing ReLU activation function
function relu(x)
    return max.(0, x)
end

function (cl::ConvLayer)(input::Array{Float32,3})
    cl.last_input = copy(input)  # Store the original input for use in the backward pass

    # Perform convolution
    conv_output = conv2d(input, cl.weights, cl.stride, cl.padding)

    # Add bias (broadcasting addition across channels)
    for c in axes(conv_output, 3)
        conv_output[:, :, c] .+= cl.biases[c]
    end

    # Apply ReLU activation function
    return relu(conv_output)
end

function init_conv_layer(kernel_height::Int, kernel_width::Int, input_channels::Int, output_channels::Int, stride::Int, padding::Int)
    weights = 0.01f0 * randn(Float32, kernel_height, kernel_width, input_channels, output_channels)  # Adjusted for multiple filters
    biases = zeros(Float32, output_channels)  # One bias per output channel
    return ConvLayer(weights, biases, stride, padding, nothing)
end

function backward_pass(cl::ConvLayer, grad_output::Array{Float32,3})
    input = cl.last_input
    (height, width, channels) = size(input)
    (kernel_height, kernel_width, _, num_kernels) = size(cl.weights)
    stride, padding = cl.stride, cl.padding

    # Initialize gradients
    grad_input = zeros(Float32, size(input))
    grad_weights = zeros(Float32, size(cl.weights))
    grad_biases = zeros(Float32, size(cl.biases))

    # Prepare padded input and gradients for input
    padded_height = height + 2 * padding
    padded_width = width + 2 * padding
    padded_input = zeros(Float32, padded_height, padded_width, channels)
    padded_input[padding+1:padding+height, padding+1:padding+width, :] = input
    padded_grad_input = zeros(Float32, size(padded_input))

    # Calculate gradients
    for k in 1:num_kernels
        for h in 1:stride:padded_height-kernel_height+1
            for w in 1:stride:padded_width-kernel_width+1
                h_out = div(h - 1, stride) + 1
                w_out = div(w - 1, stride) + 1

                # Extract the patch corresponding to the current output pixel
                patch = padded_input[h:h+kernel_height-1, w:w+kernel_width-1, :]

                # Update gradients
                if h_out <= size(grad_output, 1) && w_out <= size(grad_output, 2)
                    grad_bias = grad_output[h_out, w_out, k]
                    grad_biases[k] += grad_bias
                    grad_weights[:, :, :, k] += patch * grad_bias
                    padded_grad_input[h:h+kernel_height-1, w:w+kernel_width-1, :] += cl.weights[:, :, :, k] * grad_bias
                end
            end
        end
    end

    # Remove padding from grad_input
    grad_input .= padded_grad_input[padding+1:end-padding, padding+1:end-padding, :]
    return grad_input #, grad_weights, grad_biases
end


end
