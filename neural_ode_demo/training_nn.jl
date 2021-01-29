"""
Simple examples of training neural networks and learning
"""


using Flux, Plots

#------- linear regression step

A = rand(2, 5)
b = rand(2)

model(x) = A*x .+ b


function l2_loss(x,y)
    y_hat = model(x)
    sum((y .- y_hat).^2)
end

#We can take gradient of loss and apply gradient descent
x,y = rand(5), rand(2)
println("loss1 = ", l2_loss(x,y))
gs = gradient(() -> l2_loss(x,y), params(A, b))

display(gs[A])

println("A1 =        ")
display(A)
A_grad = gs[A]


A = A - 0.05 .* A_grad
println("A2 =        ")
display(A)
println("loss2 = ", l2_loss(x,y))

epochs = 500
x,y = rand(5), rand(2)
lr = 0.01;
for k = 1 : epochs
    global A, b
    gs = gradient(() -> l2_loss(x,y), params(A, b))
    A_grad = gs[A]
    b_grad = gs[b]

    A = A - lr .* A_grad
    b = b - lr .* b_grad

    println("loss = ", l2_loss(x,y))
end



#------------Neural networks

#Activation functions
sig(x) = 1/(1 + exp(-x))

tan_h(x) = (exp(x) - exp(-x))/(exp(x) + exp(-x))

pn1 = plot(x -> tan_h(x), -2*pi, 2*pi)
display(pn1)
pn2 = plot(x -> sig(x), -8, 8)
display(pn2)


W1 = randn(3, 5);
b1 = randn(3);
layer1(x) = W1 * x .+ b1

W2 = randn(2, 3)
b2 = randn(2);
layer2(x) = W2 * x .+ b2

nn_model(x) = layer2(sig.(layer1(x)))

println(nn_model(rand(5)))


#lets use these ideas to try and fit a function with a NN:

x_data = 0.0 : 0.1 : 2.0 * pi
y_gt = sin.(x_data)

height = 2
w1 = randn(height, 1)
b1 = randn(height)
L1(x) = w1 * x .+ b1

w2 = randn(1, height)
b2 = randn(1)
L2(x) = w2 * x .+ b2

nn_model2(x) = L2(sig.(L1(x)))

function Loss(x, y)
    y_hat = nn_model2(x)
    sum((y .- y_hat).^2)
end

# for k = 1 : 300
"""
fix grad
"""
#     grads = gradient(() -> Loss(x, y), params(w1, w2, b1, b2))
#
#     w1_grad = gs[w1]
#     b1_grad = gs[b1]
#     w2_grad = gs[w2]
#     b2_grad = gs[b2]
#
#     w1 = w1 - lr .* w1_grad
#     b1 = b1 - lr .* b1_grad
#     w2 = w2 - lr .* w2_grad
#     b2 = b2 - lr .* b2_grad
#
#     println("loss = ", Loss(y_gt, y_pred))
# end
