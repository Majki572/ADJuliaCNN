function max_pool_m(input::Array{Float32,2}, pool_size::Tuple{Int,Int})
    # Define the pool size (height and width)
    pool_height, pool_width = pool_size

    # Calculate the size of the output feature map
    output_height = size(input, 1) ÷ pool_height
    output_width = size(input, 2) ÷ pool_width

    # Initialize the output array with zeros
    output = zeros(Float32, output_height, output_width)

    # Perform max pooling
    for i in 1:output_height
        for j in 1:output_width
            # Define the region to pool over
            row_start = (i - 1) * pool_height + 1
            row_end = row_start + pool_height - 1
            col_start = (j - 1) * pool_width + 1
            col_end = col_start + pool_width - 1

            # Extract the block and find the maximum value
            pool_block = input[row_start:row_end, col_start:col_end]
            output[i, j] = maximum(pool_block)
        end
    end

    return output
end
