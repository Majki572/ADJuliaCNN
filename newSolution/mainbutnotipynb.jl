using Statistics

include("ConvolutionModule.jl")
include("PoolingModule.jl")
include("FlattenModule.jl")
include("DenseModule.jl")

include("MNISTDataLoader.jl")
include("LossAndAccuracy.jl")
include("NetworkHandlers.jl")

using .ConvolutionModule, .PoolingModule, .MNISTDataLoader, .FlattenModule, .DenseModule

# Load and preprocess the data
train_features, train_labels = MNISTDataLoader.load_data(:train)
train_x, train_y = MNISTDataLoader.preprocess_data(train_features, train_labels; one_hot=true)

# Create batches
batch_size = 100
train_data = MNISTDataLoader.batch_data((train_x, train_y), batch_size; shuffle=true)
# input_image = Float64.(input_image)

# Initialize layers
conv_layer1 = ConvolutionModule.init_conv_layer(3, 3, 1, 6, 1, 0)
pool_layer1 = PoolingModule.init_pool_layer(2, 2, 2)
conv_layer2 = ConvolutionModule.init_conv_layer(3, 3, 6, 16, 1, 0)
pool_layer2 = PoolingModule.init_pool_layer(2, 2, 2)
flatten_layer = FlattenModule.FlattenLayer()
dense_layer1 = DenseModule.init_dense_layer(400, 84, DenseModule.relu, DenseModule.relu_grad)  # Adjusted to correct input size
dense_layer2 = DenseModule.init_dense_layer(84, 10, DenseModule.identity, DenseModule.identity_grad)

# Workaround because of namespaces...
function backward_pass_master(network, grad_loss)
    for layer in reverse(network)
        if isa(layer, ConvolutionModule.ConvLayer)
            grad_loss = ConvolutionModule.backward_pass(layer, grad_loss)

        elseif isa(layer, PoolingModule.MaxPoolLayer)
            grad_loss = PoolingModule.backward_pass(layer, grad_loss)

        elseif isa(layer, DenseModule.DenseLayer)
            grad_loss = DenseModule.backward_pass(layer, grad_loss)

        elseif isa(layer, FlattenModule.FlattenLayer)
            grad_loss = FlattenModule.backward_pass(layer, grad_loss)
        else
            println("No backward pass defined for layer type $(typeof(layer))")
        end
    end
    return grad_loss
end

function update_weights(network, learning_rate)
    for layer in network
        if isa(layer, DenseModule.DenseLayer) || isa(layer, ConvolutionModule.ConvLayer)
            layer.weights .-= learning_rate * layer.grad_weights
            layer.biases .-= learning_rate * layer.grad_biases
            # Reset gradients after update
            layer.grad_weights .= 0
            layer.grad_biases .= 0
        end
    end
end

# Assemble the network
network = (conv_layer1, pool_layer1, conv_layer2, pool_layer2, flatten_layer, dense_layer1, dense_layer2)

using .NetworkHandlers, .LossAndAccuracy
epochs = 3

for epoch in 1:epochs
    accumulated_accuracy_epoch = 0.0
    accumulated_accuracy_batch = 0.0
    for i in 1:size(train_x, 4)
        input = train_x[:, :, :, i]
        target = train_y[:, i]

        # Forward pass
        output = NetworkHandlers.forward_pass_master(network, input)

        # Calculate loss, accuracy, and its gradient
        loss, accuracy, grad_loss = LossAndAccuracy.loss_and_accuracy(output, target)
        accumulated_accuracy_epoch += accuracy
        accumulated_accuracy_batch += accuracy

        # if(i % 100 == 0)
        #     println("Loss: ", loss)
        #     println("Accuracy: ", round(accumulated_accuracy_batch / 100, digits=2))
        #     accumulated_accuracy_batch = 0.0
        # end

        if (i % 10000 == 0)
            println("i ", i)
        end

        if mod(i, batch_size) == 0
            update_weights(network, 0.01)  # Learning rate
        end

        # Backward pass
        backward_pass_master(network, grad_loss)
    end
    println("Epoch $(epoch) done")
    println("Accuracy: ", round(accumulated_accuracy_epoch / size(train_x, 4), digits=2))
    accumulated_accuracy_epoch = 0.0

    update_weights(network, 0.01)
end
