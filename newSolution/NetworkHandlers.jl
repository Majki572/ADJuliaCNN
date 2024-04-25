module NetworkHandlers

function forward_pass(net, input)
    for layer in net
        input = layer(input)  # Works for both conv and pooling layers
        println("Output dimensions after layer $(typeof(layer)): ", size(input))
    end
    return input
end

function backward_pass(network, grad_loss)
    # Assuming each layer has a method to handle its own backward pass and update
    for layer in reverse(network)
        grad_loss = layer.backward(grad_loss)  # Pass the gradient back and get new gradient
    end
end

end