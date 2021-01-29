
using Zygote, ForwardDiff
using Plots

# Below is the taylor series for sin. the derivative of this should approximate
#cos.
# t = 0.0
# sign = -1.0

function s(x)
    ts = 0.0
    sign = -1.0
    # global t, sign
    for i in 1:19
        if isodd(i)
            newterm = x^i/factorial(i)
            abs(newterm)<1e-8 && return t
            # println("i=",i)
            sign = -sign
            ts += sign * newterm
        end
    end
    return ts
end


g = Zygote.gradient(s, 1.0)

display(g[1])

display(cos(1.0))

G(x) = Zygote.gradient(s, x)

M = 200
x_grid = 4*pi .* rand(M) .- 2*pi

# g_data = G.(x_grid)
# print()
g_data = zeros(M)
err = zeros(M)
for i in 1 : M
    g_data[i] = G(x_grid[i])[1]
    err[i] = g_data[i] - cos(x_grid[i])
end

# scatter(x_grid, err)

scatter(x_grid, s.(x_grid))
scatter!(x_grid, g_data)
plot!(x -> cos(x), -2*pi, 2*pi)
plot!(x -> sin(x), -2*pi, 2*pi)



# f(x) = mod(x, 2*pi)
# g3 = Zygote.gradient(f, 2*pi+0.001)
#
# println("g3 = ", g3[1])
