module NetworkHandlers

include("ConvolutionModule.jl")  # Load the module
include("PoolingModule.jl")  # Load the module
include("FlattenModule.jl")
include("DenseModule.jl")

using .ConvolutionModule, .PoolingModule, .FlattenModule, .DenseModule


function forward_pass_master(net, input)
    for layer in net
        input = layer(input)  # Works for both conv and pooling layers
        println("Output dimensions after layer $(typeof(layer)): ", size(input))
    end
    return input
end

function backward_pass_master(network, grad_loss)
    for layer in reverse(network)

        layer_string = string(typeof(layer))
        println(layer_string)
        if (cmp(layer_string, string(ConvolutionModule.ConvLayer)) == 0)
            print("YESSSS")
            grad_loss = ConvolutionModule.backward_pass(layer, grad_loss, layer.last_input)

        elseif (cmp(layer_string, string(PoolingModule.MaxPoolLayer)) == 0)
            grad_loss = PoolingModule.backward_pass(layer, grad_loss, layer.max_indices)

        elseif (cmp(layer_string, string(DenseModule.DenseLayer)) == 0)
            grad_loss = DenseModule.backward_pass(layer, grad_loss)

        elseif (cmp(layer_string, string(FlattenModule.FlattenLayer)) == 0)
            grad_loss = FlattenModule.backward_pass(layer, grad_loss)

        else
            println("No backward pass defined for layer type $(typeof(layer))")
        end
    end
    return grad_loss
end

end