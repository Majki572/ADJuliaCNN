module ConvolutionModule

struct ConvLayer
    weights::Array{Float64,4}  # Updated to 4D to handle multiple filters
    biases::Array{Float64,1}   # Biases, one per output channel
    stride::Int
    padding::Int
end

function conv2d(input::Array{Float64,3}, kernels::Array{Float64,4}, stride::Int, padding::Int)
    # Input dimensions
    (height, width, channels) = size(input)

    # Kernel dimensions
    (kernel_height, kernel_width, _, num_kernels) = size(kernels)

    # Output dimensions
    out_height = div(height - kernel_height + 2 * padding, stride) + 1
    out_width = div(width - kernel_width + 2 * padding, stride) + 1

    # Apply padding
    padded_input = zeros(Float64, height + 2 * padding, width + 2 * padding, channels)
    padded_input[padding+1:padding+height, padding+1:padding+width, :] = input

    # Output initialization
    output = zeros(Float64, out_height, out_width, num_kernels)  # Adjust for multiple output channels

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

function (cl::ConvLayer)(input::Array{Float64,3})
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
    weights = 0.01 * randn(kernel_height, kernel_width, input_channels, output_channels)  # Adjusted for multiple filters
    biases = zeros(output_channels)  # One bias per output channel
    return ConvLayer(weights, biases, stride, padding)
end

end
