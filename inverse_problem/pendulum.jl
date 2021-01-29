"""
In this file we develop a differentiable programming scheme for learnig physical
  paramters and unknown functions from data for the pendulum problem

∂_t x = v, ∂_t v = -sinx - 1/5v + F = f

may extend to two paramters and finding the sin function
∂_t x = v, ∂_t v = -sinx - αv + F

"""

using Plots, Zygote

N = 12000;
h = 10/N;
Y = [0., 3., 0., 0.];
x = Y[1:2]
println(x)
println(Y)

function f(x, F)
    return [x[2], -sin(x[1]) - 0.2*x[2] + F]
end


#RK4 method
function RK4(f, x, F)
    for n in 1 : N
        k1 = h * f(x, F);
        k2 = h * f(x + 0.5 * k1, F);
        k3 = h * f(x + 0.5 * k2, F);
        k4 = h * f(x + k3, F);
        x += (k1 + 2. * k2 + 2. * k3 + k4)/6
    end
    return x
end


function l2_loss(f, x, F, y_gt)
    y_pred = RK4(f, x, F)
    return sum((y_pred .- y_gt).^2)
end


F_gt = 0.1;
y_gt = RK4(f, x, F_gt)
display(y_gt)

F = 0.2 * rand()
y_pred = RK4(f, x, F)

# display(y_pred)
# println((y_pred .- y_gt).^2))

∂_F = gradient(F -> l2_loss(f, x, F, y_gt), F)[1]

println("∂M_∂F = ", ∂_F)
# display(∂_F)



function training_autograd(n_epchs, sample_rate, f, x, y_gt)
    F_hat = rand()
    F_track = zeros(round(Int, n_epchs/sample_rate))
    lr = 1e-5
    ii = 1;
    for k ∈ 1 : n_epchs
        ∂_F = gradient(F_hat -> l2_loss(f, x, F_hat, y_gt), F_hat)[1]
        F_hat = F_hat - lr * ∂_F
        if mod(k, sample_rate) == 0
            F_track[ii] = F_hat
            ii += 1;
            println("iteration  ", k)
        end
    end
    return F_track
end



sample_rate = 2;
lr = 1e-5
n_epchs = 200

F_out = training_autograd(n_epchs, sample_rate, f, x, y_gt)
F_gt_data = F_gt * ones(size(F_out))
# println(F_out)
plt = plot(F_out, label="F_hat", color="green")
plot!(F_gt_data, label="F_gt", color="green", linestyle=:dash)
title!("learning external forcing F with autograd")
xlabel!("epochs")
ylabel!("F_hat")
display(plt)



























#-------Analytic method

# function ℱ(x, F)
#     return [x[2], -sin(x[1]) - 0.2 * x[2] + F,
#             x[4], -cos(x[1]) * x[3] - 0.2 * x[4] + 1.];
# end
#

    # function ∂x_loss(x, v)
    #     return x/sqrt(x^2 + v^2)
    # end
    #
    # function ∂v_loss(x, v)
    #     return v/sqrt(x^2 + v^2)
    # end
    #
    #
    # function training_algorithm(n_epchs, sample_rate, x_gt, v_gt, lr, Y, ℱ)
    #     F_hat = rand()
    #     F_track = zeros(round(Int, n_epchs/sample_rate))
    #     ii = 1;
    #     for k ∈ 1 : n_epchs
    #         Y_hat = RK4(ℱ, Y, F_hat)
    #         x_hat = Y_hat[1]
    #         v_hat = Y_hat[2]
    #         h1_hat = Y_hat[3]
    #         h2_hat = Y_hat[4]
    #         gradxL = ∂x_loss(x_hat - x_gt, v_hat - v_gt) * h1_hat
    #         gradvL = ∂v_loss(x_hat - x_gt, v_hat - v_gt) * h2_hat
    #         F_hat = F_hat - lr * (gradxL + gradvL)
    #         if mod(k, sample_rate) == 0
    #             F_track[ii] = F_hat
    #             ii += 1;
    #         end
    #     end
    #     return F_track
    # end
    #
    # F_gt = 0.1
    # xv = RK4(f, x, F_gt)
    # x_gt = xv[1]
    # v_gt = xv[2];
    # println(x_gt, " ", v_gt)
    #
    # n_epchs = 4000;
    # sample_rate = 1;
    # lr = 1e-5
    #
    # F_out = training_algorithm(n_epchs, sample_rate, x_gt, v_gt, lr, Y, ℱ)
    # F_gt_data = F_gt * ones(size(F_out))
    # # println(F_out)
    # plt = plot(F_out, label="F_hat", color="green")
    # plot!(F_gt_data, label="F_gt", color="green", linestyle=:dash)
    # title!("learning external forcing F")
    # xlabel!("epochs")
    # ylabel!("F_hat")
    # display(plt)























#--------------------Tests


# F = 0.1113
# xv = RK4(f, x, F)
# println(xv)

# K = 10000
# xv_track = zeros(K, 2)
#
#
# F = -0.112
# dF = 0.00001
# for k in 1 : K
#     global F
#     xv_track[k, :] .= RK4(x, F)
#     F += dF
# end
#
# plot_xv(xv_track[:, 1], xv_track[:, 2])


# function plot_xv(x, v)
#     display(scatter(x, v))
# end
#
