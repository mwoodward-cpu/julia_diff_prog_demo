"""
simple examples of diff programming using Flux
"""

using Flux, Plots

f(x) = 3x^2 + 2x + 1;
∂f(x) = 6x + 2;

df(x) = gradient(f, x)[1];


x_data = -1. : 0.1 : 1.
f_data = f.(x_data)
∂f_data = ∂f.(x_data)
df_data = df.(x_data)

p1 = plot(x_data, f_data, label="f")
plot!(x_data, ∂f_data, label="∂f")
scatter!(x_data, df_data, label="Δf")

display(p1)


 
