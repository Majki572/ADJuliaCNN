module PoolingModule

export MaxPoolLayer, init_pool_layer, apply_pooling

struct MaxPoolLayer
    pool_height::Int
    pool_width::Int
    stride::Int
end

function init_pool_layer(pool_height::Int, pool_width::Int, stride::Int)
    MaxPoolLayer(pool_height, pool_width, stride)
end

function apply_pooling(layer::MaxPoolLayer, input::Array{Float64,3})
    # Input dimensions
    (input_height, input_width, num_channels) = size(input)

    # Calculate output dimensions
    output_height = div(input_height - layer.pool_height, layer.stride) + 1
    output_width = div(input_width - layer.pool_width, layer.stride) + 1

    # Output initialization
    output = zeros(Float64, output_height, output_width, num_channels)

    for c in 1:num_channels
        for h in 1:layer.stride:input_height-layer.pool_height+1
            for w in 1:layer.stride:input_width-layer.pool_width+1
                window = input[h:h+layer.pool_height-1, w:w+layer.pool_width-1, c]
                output[div(h - 1, layer.stride)+1, div(w - 1, layer.stride)+1, c] = maximum(window)
            end
        end
    end

    return output
end

end
