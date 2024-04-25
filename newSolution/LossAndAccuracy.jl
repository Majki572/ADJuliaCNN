module LossAndAccuracy
using Statistics: mean  # Standard library

export loss_and_accuracy

function softmax(x)
    exp_x = exp.(x .- maximum(x, dims=1))  # Subtract max for numerical stability
    return exp_x ./ sum(exp_x, dims=1)
end

function cross_entropy_loss_with_gradient(predictions, targets)
    probabilities = softmax(predictions)
    loss = -mean(sum(targets .* log.(probabilities), dims=1))
    gradient = probabilities - targets  # derivative of cross-entropy loss
    return loss, Float32.(gradient)
end

function one_cold(encoded)
    return [argmax(vec) for vec in eachcol(encoded)]
end

function loss_and_accuracy(ŷ, y)
    loss, grad = cross_entropy_loss_with_gradient(ŷ, y)
    # Convert predictions and true labels from one-hot to class indices
    pred_classes = one_cold(ŷ)
    true_classes = one_cold(y)
    acc = round(100 * mean(pred_classes .== true_classes); digits=2)  # Calculate accuracy
    return loss, acc, grad
end

end

