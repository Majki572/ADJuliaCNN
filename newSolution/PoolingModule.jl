module PoolingModule

export MaxPoolLayer, init_pool_layer, apply_pooling, maxpool_backward

struct MaxPoolLayer
    pool_height::Int
    pool_width::Int
    stride::Int
end

function init_pool_layer(pool_height::Int, pool_width::Int, stride::Int)
    MaxPoolLayer(pool_height, pool_width, stride)
end

function apply_pooling(layer::MaxPoolLayer, input::Array{Float32,3})
    (input_height, input_width, num_channels) = size(input)
    output_height = div(input_height - layer.pool_height, layer.stride) + 1
    output_width = div(input_width - layer.pool_width, layer.stride) + 1

    output = zeros(Float32, output_height, output_width, num_channels)
    max_indices = Array{Tuple{Int,Int},3}(undef, output_height, output_width, num_channels)

    for c in 1:num_channels
        for h in 1:layer.stride:input_height-layer.pool_height+1
            for w in 1:layer.stride:input_width-layer.pool_width+1
                window = input[h:h+layer.pool_height-1, w:w+layer.pool_width-1, c]
                max_value = maximum(window)
                output_idx_h = div(h - 1, layer.stride) + 1
                output_idx_w = div(w - 1, layer.stride) + 1
                output[output_idx_h, output_idx_w, c] = max_value
                idx = findfirst(isequal(max_value), window)
                max_indices[output_idx_h, output_idx_w, c] = (h + idx[1] - 2, w + idx[2] - 2)
            end
        end
    end

    return output, max_indices
end

function (layer::MaxPoolLayer)(input::Array{Float32,3})
    output, _ = apply_pooling(layer, input)
    return output
end

function backward_pass(layer::MaxPoolLayer, grad_output::Array{Float32,3})
    # Assume grad_output is reshaped correctly by FlattenLayer's backward_pass
    grad_input = zeros(Float32, calculate_input_dimensions(layer, size(grad_output)))

    for c in 1:size(grad_output, 3)
        for h in 1:size(grad_output, 1)
            for w in 1:size(grad_output, 2)
                max_h, max_w = layer.max_indices[h, w, c]
                grad_input[max_h, max_w, c] += grad_output[h, w, c]
            end
        end
    end

    return grad_input
end

# ------------------ DEPRECATED ------------------
# function backward_pass(layer::MaxPoolLayer, grad_output::Array{Float32,3}, max_indices)
#     (output_height, output_width, num_channels) = size(grad_output)
#     (input_height, input_width, _) = size(max_indices) * layer.stride

#     grad_input = zeros(Float32, input_height, input_width, num_channels)

#     for c in 1:num_channels
#         for h in 1:output_height
#             for w in 1:output_width
#                 max_h, max_w = max_indices[h, w, c]
#                 grad_input[max_h, max_w, c] += grad_output[h, w, c]
#             end
#         end
#     end

#     return grad_input
# end

end
