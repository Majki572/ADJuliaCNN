module MNISTDataLoader

using MLDatasets, Random

export load_data, preprocess_data, one_hot_encode, batch_data

function load_data(split::Symbol)
    data = MLDatasets.MNIST(split=split)
    return data.features, data.targets
end

function preprocess_data(features, targets; one_hot::Bool=true)
    # Ensure features are in the correct shape, e.g., (28, 28, 60000)
    # Reshape features to add a trivial channel dimension and organize into individual samples
    if ndims(features) == 3
        num_images = size(features, 3)
        x4dim = reshape(features, 28, 28, 1, num_images)
    else
        throw(DimensionMismatch("Expected features to have 3 dimensions"))
    end

    yhot = one_hot ? one_hot_encode(targets, 0:9) : targets
    return x4dim, yhot
end


function one_hot_encode(targets, classes)
    one_hot = zeros(Int, length(classes), length(targets))
    for (i, class) in enumerate(classes)
        filter_indices = findall(x -> x == class, targets)
        one_hot[i, filter_indices] .= 1
    end
    return one_hot
end

function batch_data(data, batch_size::Int; shuffle::Bool=true)
    x, y = data
    indices = 1:size(x, 4)
    if shuffle
        indices = Random.shuffle(indices)
    end
    return [(x[:, :, :, idx], y[:, idx]) for idx in Iterators.partition(indices, batch_size)]
end

end
