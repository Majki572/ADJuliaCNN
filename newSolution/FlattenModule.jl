module FlattenModule

export FlattenLayer, backward_pass

mutable struct FlattenLayer
    input_shape::Union{Nothing,Tuple{Int,Int,Int}}
    FlattenLayer() = new(nothing)  # Default constructor sets input_shape to nothing
end

function (layer::FlattenLayer)(input)
    if ndims(input) == 3
        layer.input_shape = size(input)
    else
        error("Input to FlattenLayer must be a 3D or 4D array.")
    end
    return reshape(input, :, size(input, 4))
end

function backward_pass(layer::FlattenLayer, grad_output::Array{Float32,2})
    if isnothing(layer.input_shape)
        error("Input shape must be set during the forward pass before calling backward_pass.")
    end
    return reshape(grad_output, layer.input_shape)
end

end
