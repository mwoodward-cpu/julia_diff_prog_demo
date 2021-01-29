
using Zygote
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
            abs(newterm)<1e-8 && return ts
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

M = 1000
x_grid = 4*pi .* rand(M) .- 2*pi

# g_data = G.(x_grid)
# print()
g_data = zeros(M)
err = zeros(M)
for i in 1 : M
    g_data[i] = G(x_grid[i])[1]
    err[i] = g_data[i] - cos(x_grid[i])
end

p1 = scatter(x_grid, err)
display(p1)

p2 = scatter(x_grid, s.(x_grid))
scatter!(x_grid, g_data)
plot!(x -> cos(x), -2*pi, 2*pi)
plot!(x -> sin(x), -2*pi, 2*pi)
display(p2)


# f(x) = mod(x, 2*pi)
# g3 = Zygote.gradient(f, 2*pi+0.001)
#
# println("g3 = ", g3[1])


#------------------Cubic spline

h = 0.5
sigma = (10. / (7. * pi * h * h));

function W(r)
  q = r / h;   if (q > 2.)   return 0.;   end
  if (q > 1.)   return (sigma * (2. - q)^3 / 4.);   end
  return (sigma * (1. - 1.5 * q * q * (1. - q / 2.)));
end

# H(r) = (d W / d r) / r
function dW_dr(r)
  q = r / h;   if (q > 2.)   return 0.;   end
  if (q > 1.)   return (-3. * sigma * (2. - q)^2 / (4. * h));   end
  return (sigma * r * (-3. + 9. * q / 4.) / h^2);
end


g2 = Zygote.gradient(W, 1.0)

display(g2[1])

G2(x) = Zygote.gradient(W, x)

M = 100
r_grid = 2 .* h .* rand(M)


g2_data = zeros(M)
err = zeros(M)
for i in 1 : M
    g2_data[i] = G2(r_grid[i])[1]
    err[i] = g2_data[1] - dW_dr(r_grid[1])
end


p3 = scatter(r_grid, W.(r_grid), label = "W")
scatter!(r_grid, g2_data, color = "red", label = "W.gradient")
plot!(x -> dW_dr(x), 0, 2*h, color = "black", label = "∂W/∂r", linewidth=2)
display(p3)

p4 = scatter(r_grid, err)
display(p4)
