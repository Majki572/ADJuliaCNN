using TensorOperations
import Random: shuffle, shuffle!, seed!
seed!(0)

include("diff.jl")
include("subviews.jl")
include("optimizers.jl")
cross_entropy_loss(y, ŷ) = sum(-y .* log.(ŷ))
dense(w, b, n::Integer, m::Integer, v, activation::Function) =
    activation(reshape(w, n, m) * reshape(v, m, 1) .+ reshape(b, n, 1))
softmax(x) = exp.(x) ./ sum(exp.(x))

function conv(kernels, input, n, m)
    filters = reshape(kernels, 3, 3, :)
    N, M = (n, m) .- 2
    _, _, K = size(filters)
    input = reshape(input, n, m)
    output = zeros(N, M, K)
    for i = 1:N
        for j = 1:M
            region = input[i:i+2, j:j+2]
            output[i, j, :] = @tensor out[k] := region[i, j] * filters[i, j, k]
        end
    end
    output
end

function maxpool(input)
    n, m, k = size(input)
    N, M = floor.(Integer, (n, m) ./ 2)
    output = similar(input, N, M, k)
    for i = 1:N
        for j = 1:M
            region = input[2(i)-1:2(i)-0, 2(j)-1:2(j)-0, :]
            output[i, j, :] = maximum(region; dims=(1, 2))
        end
    end
    output
end

fs = 32
cs = 10

parameters = zeros(3 * 3 * fs + cs * 13 * 13 * fs + cs)
wh, wo, bo = subviews(parameters, (3, 3, fs), (cs, 13 * 13 * fs), (cs))
wh .= randn(3, 3, fs) ./ 9 # to reduce variance
wo .= randn(cs, 13 * 13 * fs) ./ (13 * 13 * fs) # to reduce variance

η = 0.1
ε = 1e-9
epochs = 30
batch_size = 128
optimizer = Adagrad(η, ε, length(parameters))

function net(x, wh, wo, bo)
    x̄ = conv(wh, x, 28, 28)
    x̂ = maxpool(x̄)
    ŷ = dense(wo, bo, cs, 13 * 13 * fs, x̂, softmax)
end

function loss(x, y, wh, wo, bo)
    ŷ = net(x, wh, wo, bo)
    cross_entropy_loss(y, ŷ)
end

include("mnist.jl")
test_size = 100
train_size = 1280
data_size = train_size + test_size
train_set = shuffle(1:data_size)[1:train_size]
test_set = setdiff(1:data_size, train_set)

function dloss(x, y, wh, wo, bo)
    x, y = Variable(x), Variable(y)
    wh = Variable(convert(Array{Float64}, wh))
    wo = Variable(convert(Array{Float64}, wo))
    bo = Variable(convert(Array{Float64}, bo))
    L = loss(x, y, wh, wo, bo)
    backward(L)
    return wh |> grad |> vec, wo |> grad |> vec, bo |> grad |> vec
end

function test(parameters, test_set)
    wh, wo, bo = subviews(parameters, (3, 3, fs), (cs, 13 * 13 * fs), (cs))
    Et = zero(0.0)
    for j = test_set
        x = inputs[:, :, j]
        y = targets[:, j, :]

        Et += loss(x, y, wh, wo, bo)
    end
    return Et / length(test_set)
end

function train(parameters, train_set)
    wh, wo, bo = subviews(parameters, (3, 3, fs), (cs, 13 * 13 * fs), (cs))
    ∇E = zeros(length(parameters))
    for j = train_set
        x = inputs[:, :, j]
        y = targets[:, j, :]
        ŷ = net(x, wh, wo, bo)

        Ewh, Ewo, Ebo = dloss(x, y, wh, wo, bo)

        ∇E .+= vcat(Ewh, Ewo, Ebo)
    end
    return ∇E / length(train_set)
end

function optimize!(parameters)
    parameters .= step!(optimizer,
        p -> test(p, test_set), p -> train(p, train_set[1:batch_size]),
        parameters)
end

for i = 1:epochs
    shuffle!(train_set)
    optimize!(parameters)
    println(i, "\t", test(parameters, train_set))
    flush(stdout)
end
println("FINAL", "\t", test(parameters, train_set))