using Random

function loader_m(data; batchsize::Int=1, shuffle::Bool=true)
    x4dim = reshape(data.features, 28, 28, 1, :) # Reshape with channel dimension
    targets = data.targets
    yhot = [Int(target == i) for target in targets, i in 0:9] # One-hot encode the labels

    indices = 1:length(targets)
    if shuffle
        indices = shuffle(indices) # Shuffle indices if required
    end

    batches = [(x4dim[:, :, :, i:i+batchsize-1], yhot[:, i:i+batchsize-1]) for i in indices[1:batchsize:end]]
    return batches
end

function flatten_m(x::Array)
    return reshape(x, :)
end

function softmaxior(logits::Array{Float64})
    max_logit = maximum(logits)
    exp_logits = exp.(logits .- max_logit)
    sum_exp_logits = sum(exp_logits)
    return exp_logits / sum_exp_logits
end

function cross_entropy(predictions::Array{Float64}, targets::Array{Int})
    # Ensure numerical stability
    epsilon = 1e-12
    predictions = clamp.(predictions, epsilon, 1 - epsilon)

    # Calculate the cross-entropy loss
    N = size(targets, 1)
    loss = -sum(targets .* log.(predictions) + (1 .- targets) .* log.(1 .- predictions)) / N
    return loss
end