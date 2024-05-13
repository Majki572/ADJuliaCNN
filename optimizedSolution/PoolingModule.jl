module PoolingModule

export MaxPoolLayer, init_pool_layer, apply_pooling, maxpool_backward

mutable struct MaxPoolLayer
    pool_height::Int
    pool_width::Int
    stride::Int
    max_indices::Union{Nothing,Array{Tuple{Int,Int},3}}
    input_height::Int
    input_width::Int
    num_channels::Int
    output_height::Int
    output_width::Int
    output::Array{Float32,3}
    grad_input::Union{Nothing,Array{Float32,3}}
end

function init_pool_layer(pool_height::Int, pool_width::Int, stride::Int, inp_height::Int, inp_width::Int, inp_channels::Int)
    output_height = div(inp_height - pool_height, stride) + 1
    output_width = div(inp_width - pool_width, stride) + 1
    output = zeros(Float32, output_height, output_width, inp_channels)
    max_indices = Array{Tuple{Int,Int},3}(undef, output_height, output_width, inp_channels)

    MaxPoolLayer(pool_height, pool_width, stride, max_indices, inp_height, inp_width, inp_channels, output_height, output_width, output, nothing)
end

function apply_pooling(layer::MaxPoolLayer, input::Array{Float32,3})

    height_bounds = layer.input_height - layer.pool_height + 1
    width_bounds = layer.input_width - layer.pool_width + 1
    for c in 1:layer.num_channels
        for h in 1:layer.stride:height_bounds
            for w in 1:layer.stride:width_bounds
                @views window = input[h:h+layer.pool_height-1, w:w+layer.pool_width-1, c]
                max_value = -Inf
                max_index = (0, 0)
                size_one = size(window, 1)
                size_two = size(window, 2)

                for i in 1:size_one
                    for j in 1:size_two
                        current_value = window[i, j]
                        if current_value > max_value
                            max_value = current_value
                            max_index = (i, j)
                        end
                    end
                end

                output_idx_h = div(h - 1, layer.stride) + 1
                output_idx_w = div(w - 1, layer.stride) + 1
                layer.output[output_idx_h, output_idx_w, c] = max_value
                layer.max_indices[output_idx_h, output_idx_w, c] = (h + max_index[1] - 1, w + max_index[2] - 1)
            end
        end
    end

    return layer.output, layer.max_indices
end

function (layer::MaxPoolLayer)(input::Array{Float32,3})
    output, _ = apply_pooling(layer, input)
    return output
end

function backward_pass(layer::MaxPoolLayer, grad_output::Array{Float32,3})
    if layer.grad_input === nothing
        layer.grad_input = zeros(Float32, calculate_input_dimensions(layer, size(grad_output)...))
    end

    gi1, gi2, gi3 = size(layer.grad_input, 1), size(layer.grad_input, 2), size(layer.grad_input, 3)
    for i in 1:gi1
        for j in 1:gi2
            for k in 1:gi3
                layer.grad_input[i, j, k] = 0.0
            end
        end
    end #layer.grad_input .= 0

    for c in 1:size(grad_output, 3)
        for h in 1:size(grad_output, 1)
            for w in 1:size(grad_output, 2)
                max_h, max_w = layer.max_indices[h, w, c]
                layer.grad_input[max_h, max_w, c] += grad_output[h, w, c]
            end
        end
    end

    return layer.grad_input
end

function calculate_input_dimensions(layer::MaxPoolLayer, out_height::Int, out_width::Int, num_channels::Int)
    input_height = out_height * layer.stride + layer.pool_height - 1
    input_width = out_width * layer.stride + layer.pool_width - 1
    return (input_height, input_width, num_channels)
end


end
