module FlattenModule

export FlattenLayer

struct FlattenLayer end

function (layer::FlattenLayer)(input)
    if ndims(input) == 3
        # Assuming input as [height, width, channels], typically happens with single image (batch size 1)
        input = reshape(input, size(input)..., 1)  # Convert to 4D by adding batch dimension
    end
    return reshape(input, :, size(input, 4))
end

# The backward pass function that reshapes the gradient back to the input's original shape
function backward_pass(layer::FlattenLayer, grad_output::Array)
    return reshape(grad_output, layer.input_shape)
end

end
