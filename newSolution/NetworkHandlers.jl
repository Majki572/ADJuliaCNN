module NetworkHandlers

function forward_pass_master(net, input)
    for layer in net
        input = layer(input)  # Works for both conv and pooling layers
        println("Output dimensions after layer $(typeof(layer)): ", size(input))
    end
    return input
end

end