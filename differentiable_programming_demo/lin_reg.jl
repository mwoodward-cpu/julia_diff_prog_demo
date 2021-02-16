using Zygote, LinearAlgebra



mutable struct LinearRegression
    # These values will be implicitly learned
    weights::Matrix
    bias::Float64

    # These values will not be learned
    name::String
end

LinearRegression(nparams, name) = LinearRegression(randn(1, nparams), 0.0, name)


# y = w*X + b
function predict(model::LinearRegression, X)
    return model.weights * X .+ model.bias
end

# L2 loss:
function loss(model::LinearRegression, X, Y)
    return norm(predict(model, X) .- Y, 2)
end


# Our "ground truth" values (that we will learn, to prove that this works)
weights_gt = [1.0, 2.7, 0.3, 1.2]'
bias_gt = 0.4

# Generate a dataset of many observations
X = randn(length(weights_gt), 10000)
Y = weights_gt * X .+ bias_gt

# Add a little bit of noise to `X` so that we do not have an exact solution,
# but must instead do a least-squares fit:
X .+= 0.001.*randn(size(X))
display(X)
