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

epochs = 200
x,y = rand(5), rand(2)
for k = 1 : epochs
    global A, b
    gs = gradient(() -> l2_loss(x,y), params(A, b))
    A_grad = gs[A]
    b_grad = gs[b]

    A = A - 0.01 * A_grad
    b = b - 0.01 * b_grad

    println("loss = ", l2_loss(x,y))
end


#-------------Simple NN
