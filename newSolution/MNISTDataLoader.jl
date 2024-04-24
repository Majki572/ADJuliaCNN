module MNISTDataLoader

using MLDatasets, Random

export load_dataset, preprocess_data, one_hot_encode, batch_data

function load_dataset(split::Symbol)
    # Load MNIST data for the specified split (train or test)
    return MLDatasets.MNIST(split=split)
end

function preprocess_data(features, targets; one_hot::Bool=true)
    # Reshape features to add a trivial channel dimension and apply one-hot encoding if specified
    x4dim = reshape(features, 28, 28, 1, :)
    yhot = one_hot ? one_hot_encode(targets, 0:9) : targets
    return x4dim, yhot
end

function one_hot_encode(targets, classes)
    # Create a one-hot encoded matrix for the targets based on specified classes
    one_hot = zeros(Int, length(classes), length(targets))
    for (i, class) in enumerate(classes)
        filter_indices = findall(x -> x == class, targets)
        one_hot[i, filter_indices] .= 1
    end
    return one_hot
end

function batch_data(data, batch_size::Int; shuffle::Bool=true)
    # Split data into batches, optionally shuffling before partitioning
    x, y = data
    indices = 1:size(x, 4)
    if shuffle
        indices = shuffle(indices)
    end
    return [(x[:, :, :, idx], y[:, idx]) for idx in Iterators.partition(indices, batch_size)]
end

end
