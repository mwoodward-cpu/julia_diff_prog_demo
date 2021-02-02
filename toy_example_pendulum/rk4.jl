"""
Consider the problem of determining the parameter F
from data:

∂_t x = v, ∂_t v = -sinx - 1/5v + F

may extend to two paramters
∂_t x = v, ∂_t v = -sinx - αv + F

"""

function f(x)
    return [x[2], -sin(x[1]) - 0.2*x[2] + F]
end


#RK4 method
for n in 1 : N
    global x
    k1 = h * f(x);
    k2 = h * f(x + 0.5 * k1);
    k3 = h * f(x + 0.5 * k2);
    k4 = h * f(x + k3);
    x += (k1 + 2. * k2 + 2. * k3 + k4)/6
end
