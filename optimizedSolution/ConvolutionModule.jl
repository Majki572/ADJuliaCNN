module ConvolutionModule
using Random

mutable struct ConvLayer
    weights::Array{Float32,4}
    biases::Array{Float32,1}
    grad_weights::Array{Float32,4}
    grad_biases::Array{Float32,1}
    stride::Int
    padding::Int
    last_input::Array{Float32,3}  # Adding this field to store the input
    # Caches for optimization -6GiB
    height::Int
    width::Int
    channels::Int
    kernel_height::Int
    kernel_width::Int
    num_kernels::Int
    out_height::Int
    out_width::Int
    output::Array{Float32,3}
    padded_input::Array{Float32,3}
    grad_input::Array{Float32,3}
end

function forward(input::Array{Float32,3}, cl::ConvLayer, kernels::Array{Float32,4}, stride::Int, padding::Int)

    # Apply padding
    if padding > 0
        fill!(cl.padded_input, 0)
        cl.padded_input[cl.padding+1:cl.padding+cl.height, cl.padding+1:cl.padding+cl.width, :] = input
    else
        cl.padded_input = input
    end


    for k in 1:size(cl.output, 3)
        for i in 1:size(cl.output, 1)
            for j in 1:size(cl.output, 2)
                cl.output[i, j, k] = 0.0
            end
        end
    end #fill!(cl.output, 0)

    height_bounds = cl.height - cl.kernel_height + 1 + 2 * padding
    width_bounds = cl.width - cl.kernel_width + 1 + 2 * padding
    # Perform convolution for each filter
    for k in 1:cl.num_kernels
        kernel = kernels[:, :, :, k]
        for h in 1:stride:height_bounds
            for w in 1:stride:width_bounds
                @views patch = cl.padded_input[h:h+cl.kernel_height-1, w:w+cl.kernel_width-1, :]

                sum_val = 0.0

                for kh in 1:cl.kernel_height
                    for kw in 1:cl.kernel_width
                        for kc in 1:cl.channels
                            sum_val += patch[kh, kw, kc] * kernel[kh, kw, kc]
                        end
                    end
                end

                cl.output[div(h - 1, stride)+1, div(w - 1, stride)+1, k] += sum_val #sum(patch .* kernel)
            end
        end
    end

    return cl.output
end

function (cl::ConvLayer)(input::Array{Float32,3})
    cl.last_input = input  # Store the original input for use in the backward pass

    # Perform convolution
    conv_output = forward(input, cl, cl.weights, cl.stride, cl.padding)

    # Add bias (broadcasting addition across channels)
    for c in axes(conv_output, 3)
        conv_output[:, :, c] .+= cl.biases[c]
    end

    # Apply ReLU activation function
    return relu(conv_output)
end

# Implementing ReLU activation function
function relu(x)
    for i in eachindex(x)
        x[i] = max(0, x[i])
    end
    return x
end

function init_conv_layer(kernel_height::Int, kernel_width::Int, input_channels::Int, output_channels::Int, stride::Int, padding::Int, seedy::Int, inp_height::Int, inp_width::Int, inp_channels::Int)

    # seed = rand(UInt32)
    Random.seed!(seedy)

    weights = 0.178 * randn(Float32, kernel_height, kernel_width, input_channels, output_channels)
    biases = zeros(Float32, output_channels)
    grad_weights = zeros(Float32, kernel_height, kernel_width, input_channels, output_channels)
    grad_biases = zeros(Float32, output_channels)
    last_input = zeros(Float32, inp_height, inp_width, inp_channels)
    # Prepare caches
    (kh, kw, _, num_kernels) = size(weights)
    out_height = div(inp_height - kh + 2 * padding, stride) + 1
    out_width = div(inp_width - kw + 2 * padding, stride) + 1
    output = zeros(Float32, out_height, out_width, num_kernels)
    grad_input = zeros(Float32, (inp_height, inp_width, inp_channels))
    padded_input = zeros(Float32, inp_height + 2 * padding, inp_width + 2 * padding, inp_channels)

    return ConvLayer(weights, biases, grad_weights, grad_biases, stride, padding, last_input, inp_height, inp_width, inp_channels, kh, kw, num_kernels, out_height, out_width, output, padded_input, grad_input)
end

function backward_pass(cl::ConvLayer, grad_output::Array{Float32,3})

    input = cl.last_input

    cl.grad_input .= 0
    # Prepare padded input and gradients for input

    if cl.padding > 0
        cl.padded_input .= 0
        cl.padded_input[cl.padding+1:cl.padding+cl.height, cl.padding+1:cl.padding+cl.width, :] = input
    else
        cl.padded_input = input
    end

    h_output = size(grad_output, 1)
    w_output = size(grad_output, 2)

    height_bounds = cl.height - cl.kernel_height + 1 + 2 * cl.padding
    width_bounds = cl.width - cl.kernel_width + 1 + 2 * cl.padding

    for k in 1:cl.num_kernels
        for h in 1:cl.stride:height_bounds
            for w in 1:cl.stride:width_bounds
                h_out = div(h - 1, cl.stride) + 1
                w_out = div(w - 1, cl.stride) + 1
                if h_out <= h_output && w_out <= w_output
                    @views patch = cl.padded_input[h:h+cl.kernel_height-1, w:w+cl.kernel_width-1, :]
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
    return Float32.(cl.grad_input) #, grad_weights, grad_biases
end


end
