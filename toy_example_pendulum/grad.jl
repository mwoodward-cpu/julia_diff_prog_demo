"""
analytic gradient
"""

function f2(x)
    global F
    return [x[2], -sin(x[1]) - 0.2 * x[2] + F,
            x[4], -cos(x[1]) * x[3] - 0.2 * x[4] + 1.];
end

N = 15000;
h = 15/N;
F = parse(Float64, ARGS[1]);
println(F)
x = [0., 3., 0., 0.];


for n in 1 : N
    global x
    k1 = h * f2(x);
    k2 = h * f2(x + 0.5 * k1);
    k3 = h * f2(x + 0.5 * k2);
    k4 = h * f2(x + k3);
    x += (k1 + 2. * k2 + 2. * k3 + k4)/6
end

println(F, " ", x[1], " ", x[2], " ", x[3], " ", x[4]);
