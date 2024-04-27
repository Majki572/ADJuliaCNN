using Statistics

include("ConvolutionModule.jl")  # Load the module
include("PoolingModule.jl")  # Load the module
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
batch_size = 100  # Define your desired batch size
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
function backward_pass_master(network, grad_loss, transition_output_pool=nothing)
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


# Assemble the network
network = (conv_layer1, pool_layer1, conv_layer2, pool_layer2, flatten_layer, dense_layer1, dense_layer2)

using .NetworkHandlers, .LossAndAccuracy
function train_epoch(network, inputs, targets, epochs)
    for epoch in 1:epochs
        for i in 1:size(inputs, 4)  # Iterate over each example
            input = inputs[:, :, :, i]
            target = targets[:, i]

            # Forward pass
            output = NetworkHandlers.forward_pass_master(network, input)

            # Calculate loss, accuracy, and its gradient
            loss, accuracy, grad_loss = LossAndAccuracy.loss_and_accuracy(output, target)

            if (i % 100 == 0)
                println("Loss: ", loss)
                println("Accuracy: ", accuracy)
            end

            # Backward pass
            backward_pass_master(network, grad_loss)
        end
        println("Epoch $(epoch) done")
    end
end

train_epoch(network, train_x, train_y, 3)