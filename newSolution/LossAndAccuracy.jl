module LossAndAccuracy
using Statistics: mean  # standard library

export calculate_loss_and_accuracy

function softmax(x)
    exp_x = exp.(x .- maximum(x, dims=1))  # Subtract max for numerical stability
    return exp_x ./ sum(exp_x, dims=1)
end

function cross_entropy_loss(predictions, targets)
    probabilities = softmax(predictions)
    return -mean(sum(targets .* log.(probabilities), dims=1))
end

function one_cold(encoded)
    return [argmax(vec) for vec in eachcol(encoded)]
end

function loss_and_accuracy(ŷ, y)
    loss = cross_entropy_loss(ŷ, y)

    # Convert predictions and true labels from one-hot to class indices
    pred_classes = one_cold(ŷ)
    true_classes = one_cold(y)

    # Calculate accuracy
    acc = round(100 * mean(pred_classes .== true_classes); digits=2)

    return (loss=loss, acc=acc)
end

end
