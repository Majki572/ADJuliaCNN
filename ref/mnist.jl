using MLDatasets

images, classes = MNIST.traindata()
inputs = images ./ 255
targets = zeros(10, 60_000)
for i = 1:60_000
    targets[classes[i]+1, i] = 1
end