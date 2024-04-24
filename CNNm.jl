function conv_m(input::Array{Float64,2}, kernel::Array{Float64,2}, stride::Int=1, pad::Int=0)
    # Apply padding to the input image
    padded_input = pad_array(input, pad)

    # Get dimensions of the padded input and the kernel
    (input_height, input_width) = size(padded_input)
    (kernel_height, kernel_width) = size(kernel)

    # Determine the output dimensions
    output_height = ((input_height - kernel_height) ÷ stride) + 1
    output_width = ((input_width - kernel_width) ÷ stride) + 1

    # Initialize the output matrix
    output = zeros(output_height, output_width)

    # Perform the convolution operation with stride
    for i in 1:stride:output_height
        for j in 1:stride:output_width
            # Extract the relevant part of the padded input image
            input_patch = padded_input[(i-1)*stride+1:(i-1)*stride+kernel_height,
                (j-1)*stride+1:(j-1)*stride+kernel_width]

            # Compute the convolution for this patch
            output[i, j] = sum(input_patch .* kernel)
        end
    end

    return output
end

function pad_array(input::Array{Float64,2}, pad::Int)
    if pad == 0
        return input
    else
        padded_height = size(input, 1) + 2 * pad
        padded_width = size(input, 2) + 2 * pad
        padded_input = zeros(padded_height, padded_width)
        padded_input[pad+1:end-pad, pad+1:end-pad] = input
        return padded_input
    end
end

function relu_m(x)
    return max.(0, x)
end
