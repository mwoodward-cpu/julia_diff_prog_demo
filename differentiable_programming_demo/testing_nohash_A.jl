"""
In this file we test the individual functions and compositions
with zygote gradient
"""



using Zygote, ForwardDiff
using Distances
using Plots
using StatsPlots

#------- Below is the taylor series for sin. the derivative of this should approximate
##Note that in place operations are ok
#cos.
# t = 0.0
# sign = -1.0
#
# function s(x)
#     # t = 0.0
#     # sign = -1.0
#     global t, sign
#     for i in 1:19
#         if isodd(i)
#             newterm = x^i/factorial(i)
#             abs(newterm)<1e-8 && return t
#             println("i=",i)
#             sign = -sign
#             t += sign * newterm
#         end
#     end
#     return t
# end
#
#
# g = Zygote.gradient(s, 1.0)
#
# display(g[1])
#
# display(cos(1.0))

#now lets test this for larger functions:



#--------- Testing automatic diff for sph
h = 0.125
c = 20
g = 7
dt = 0.30*h/c;
c0 = c^2/g
α = -0.2
beta = 0.4

power = 10

N = 2^power
D = 2

m = (2. * pi)^D / N

#initial conditions for X
X = 0.2 .* randn(N, D); V = randn(N, D); A = zeros(N, D);
rho = zeros(N);
X = mod.(X, 2*pi)

sigma = (10. / (7. * pi * h * h));
function W(r)
  global h, sigma;
  q = r / h; if (q > 2.)   return 0.;   end
  if (q > 1.)   return (sigma * (2. - q)^3 / 4.);   end
  return (sigma * (1. - 1.5 * q * q * (1. - q / 2.)));
end

function H(r) # H(r) = (d W / d r) / r
  global h, sigma;
  q = r / h; if (q > 2.)   return 0.;   end
  if (q > 1.)   return (-3. * sigma * (2. - q)^2 / (4. * h * r));   end
  return (sigma * (-3. + 9. * q / 4.) / h^2);
end

function Pres(rho)
  global g
  return (c0 * (rho^g - 1.))
end

function obtain_rho_scaler(X, i)
  """
  This computes the density at particle i by summing over j
  This seems to be working with autograd
  and it pruduces equivalent densities to original obtain_rho
  """
  global h;
  rho_s = 0.0
  for j in 1 : N
    r2 = 0.;
    for k in 1 : D
      XX = X[i, k] - X[j, k];
      if (XX > pi)
        XX -= 2. * pi;
      end
      if (XX < -pi)
        XX += 2. * pi;
      end
      r2 += XX^2;
    end
    if (r2 <= 4. * h * h)
      tmp = m * W(sqrt(r2));
      rho_s += tmp;
    end
  end
  return rho_s
end


function obtain_vec_rho(X)
  """
  From obtain_rho_scaler, this computes the vector rho over all N particles
  """
  I = 1 : (N)   #indicies over i ∈ {1, 2, ... N}
  rho_ = obtain_rho_scaler.(Ref(X), I)
  return rho_
end



As = zeros(D)
function F_ext(X, i)
  """
  external forcing for each particle i
  """
  Asx = sin(X[i, 1]) * cos(X[i, 2]);
  Asy = -cos(X[i, 1]) * sin(X[i, 2]);
  As = vcat(Asx, Asy)
end

function F_vec_ext(X,i)
  I = 1 : (N - 1)
  F_ev = F_ext.(Ref(X), I)
end



#initial conditions seeded
for i in 1 : round(Int, 2^(power/2))   for j in 1 : round(Int, 2^(power/2))
  n = round(Int, 2^(power/2)) * (i - 1) + j;
  X[n, 1] = 2. * pi * ((i - 1.) / sqrt(N) + 0.001 * (rand() - 0.5));
  X[n, 2] = 2. * pi * ((j - 1.) / sqrt(N) + 0.001 * (rand() - 0.5));
end   end
for n in 1 : N
  V[n, 1] = V[n, 2] = 0.;
end
X = mod.(X, 2*pi)




I = 1 : N
# println("ref = ", Ref(I))
# rho_ = obtain_rho_scaler.(Ref(X), I)
# rho_ = obtain_rho_scaler.(Ref(X), I)
rho_ = obtain_vec_rho(X)

display(rho_)
rho_test = zeros(N)
for n ∈ 1 : N
  rho_test[n] = obtain_rho_scaler(X, n)
end






#---------- A scalar
"""
Apply what we know for rho_scaler to obtain_A scaler

"""

function obtain_A_scalar(X, i)
  """
  This computes the Acceleration term at particle i by summing over j
  This needs work, currently it is not producing same A as obtain_A
  """

  global V, rho, h, As
  As = zeros(D)
  XX = zeros(D);
  VV = zeros(D);
  μ = 0.0
  Π = 0.0

  for j in 1 : N
    r2 = 0.;
    # XXt = X[i, :] - X[j, :];
    VVt = V[i, :] - V[j, :];
    for k in 1 : D
      XXt[k] = X[i, k] - X[j, k];
      while (XXt[k] > pi) XXt[k] -= 2. * pi; end
      while (XXt[k] < -pi) XXt[k] += 2. * pi; end
      r2 += XXt[k] * XXt[k];
    end
    if (r2 <= 4. * h * h)
      if (sum(XXt .* VVt) < 0)
        μ = h*(sum(XXt .* VVt))/(sqrt(r2) + 0.01*h^2);
        Π = (α*c*μ + beta * μ^2)/((rho[i] + rho[j])/2);
      end
      if (sum(XXt .* VVt) >= 0)
        Π = 0.0
      end
      tmp = m * (Pres(rho[j])/rho[j]^2 + Pres(rho[i])/rho[i]^2 + Π) * H(sqrt(r2));  #conservative Presure term
      # As += XXt .* tmp
      for k in 1:D
        As[k] += XXt[k] * tmp
      end
    end
  end
  return As
end


function obtain_vec_A(α, beta)
  """
  From obtain_rho_scaler, this computes the vector rho over all N particles
  """
  I = 1 : (N)   #indicies over i ∈ {1, 2, ... N}
  As_ = obtain_A_scalar.(α, beta, I)
  return hcat(As_...)'
end


#--------------- Testing As function
"""
Needs work
"""
function obtain_A()
  global X, A, rho, h, V;
  μ = 0.0
  Π = 0.0
  A = zeros(N,D)
  # for n in 1 : N # four vortices on a 2D torus
  #   A[n, 1] = sin(X[n, 1]) * cos(X[n, 2]);
  #   A[n, 2] = -cos(X[n, 1]) * sin(X[n, 2]);
  # end
  for n1 in 1 : (N - 1)   for n2 in (n1 + 1) : N
    close = true; r2 = 0.; XX = zeros(D); VV = zeros(D);
    for i in 1 : D
      XX[i] = X[n1, i] - X[n2, i];
      VV[i] = V[n1, i] - V[n2, i];
      while (XX[i] > pi) XX[i] -= 2. * pi; end
      while (XX[i] < -pi) XX[i] += 2. * pi; end
      r2 += XX[i] * XX[i];
      if (r2 > 4. * h * h) close = false; break; end
    end
    if (close)
      if (XX[1]*VV[1] + XX[2]*VV[2] < 0)
        μ = h*(XX[1]*VV[1] + XX[2]*VV[2])/(sqrt(r2) + 0.01*h^2)
        Π = m*(α*c*μ + beta * μ^2)/((rho[n1] + rho[n2])/2)
      end
      if (XX[1]*VV[1] + XX[2]*VV[2] >= 0)
        Π = 0.0
      end
      tmp = m * (Pres(rho[n1])/rho[n1]^2 + Pres(rho[n2])/rho[n2]^2 + Π) * H(sqrt(r2));  #conservative Presure term
      for i in 1 : D
        A[n1, i] -= XX[i] * (tmp);
        A[n2, i] += XX[i] * (tmp);
      end
    end
  end   end
  return A
end

rho = obtain_vec_rho(X)
obtain_A()
As = obtain_vec_A(α, beta)
display(A)
display(As)
# display(hcat(As...)')
# display(A .- As)


# p1 = density(rho, title = "rho")
# display(p1)
#
# p2 = density(rho_test, title="rho test")
# display(p2)
#
# p3 = density(rho_, title="rho test 2")
# display(p3)

# t3 = obtain_vec_rho(X)
# println(t3)

# t_sum(X) = sum(obtain_vec_rho(X))
# t_sum(X) = sum(abs2, obtain_vec_rho(X))
# g4 = Zygote.gradient(t_sum, X)
# println("vec_rho grad = ", g4)

# t2_sum(X) = sum(F_ext(X, 2))
# g5 = Zygote.gradient(t2_sum, X)
# println("ext_force = ", g5)


# t3_sum(X) = sum(obtain_rho_scaler.(Ref(X), I))
# t3_sum(X) = sum(obtain_vec_rho(X))
# g3s = Zygote.gradient(t2_sum, X)
# println(g3s[1])




#--------------- Testing rho function
"""
So far it seems that we have a working AD form for obtain_vec_rho
  which can be seen from the below tests
"""
#
# obtain_rho()
# p1 = density(rho, title = "rho")
# display(p1)
#
# p2 = density(rho_test, title="rho test")
# display(p2)
#
# p3 = density(rho_, title="rho test 2")
# display(p3)

# t3 = obtain_vec_rho(X)
# println(t3)

# t_sum(X) = sum(obtain_vec_rho(X))
# t_sum(X) = sum(abs2, obtain_vec_rho(X))
# g4 = Zygote.gradient(t_sum, X)
# println("vec_rho grad = ", g4)

# t2_sum(X) = sum(F_ext(X, 2))
# g5 = Zygote.gradient(t2_sum, X)
# println("ext_force = ", g5)


# t3_sum(X) = sum(obtain_rho_scaler.(Ref(X), I))
# t3_sum(X) = sum(obtain_vec_rho(X))
# g3s = Zygote.gradient(t2_sum, X)
# println(g3s[1])


#------------ Integrator
ii = 1
function verlet(X, V, T, c, α, beta)
  """
  computes the trajectories and velocities of lagrangian particles
  solved using SPH
  """
  for k in 1 : T
    global ii, dt, c

    rho = obtain_rho();
    A = obtain_A();

    V += 0.5 * dt * A;
    X += dt * V;
    X = mod.(X, 2*pi)

    rho = obtain_rho();
    A = obtain_A();

    V += 0.5 * dt * A;
    t += dt;

    vel[k + 1, :, :] = V
    pos[k + 1, :, :] = X
    println("time step:", ii)
    ii += 1
  end
  return pos, vel
end

# t4_sum(X) = sum(verlet(X, V, 3, c, α, beta))
# g4s = Zygote.gradient(t4_sum, X)
# println(g4s[1])
#









#
#
#------Pairwise dist
# dist = Euclidean()
#
# function obtain_rho_array(X)
#   global rho, h;
#   R = pairwise(dist, X, X, dims=1)
#   R = mod.(R, 2*pi)
#   Wmat = W.(R)
#   rho = m .* sum(Wmat, dims=2)
#   return rho
# end
# #
# # Trho(X) = sum(obtain_rho_array(X))
# # rho_arrar = obtain_rho_array(X)
# # println("rho_arrar = ", rho_arrar)
# #
# # g4 = Zygote.gradient(Trho, mod.(randn(N,D), 2*pi))
# # display(g4[1])
#
# # Xtest = [3*pi 2; 5*pi 7]
# # display(Xtest)
# # mod_array = 2*pi .* ones(2,2)
# # display(mod.(Xtest, 2*pi))
# # # rho = obtain_rho(X)
# #
# # Total2(X) = sum(obtain_rho(X))
# # display(Total2(X))
# #
# #
# # g2 = Zygote.gradient(Total2, randn(N,D))
# #
# #
# # display(g2)
#
#
#
#
#
#
# function Pres(rho)
#   return (c * (rho^g - 1.))
# end
#
#
# function obtain_A(α, beta)
#   global V, rho, h, X
#   A = zeros(N,D)
#   μ = 0.0
#   Π = 0.0
#  # α = -1
#   for n in 1 : N # four vortices on a 2D torus
#     A[n, 1] = sin(X[n, 1]) * cos(X[n, 2]);
#     A[n, 2] = -cos(X[n, 1]) * sin(X[n, 2]);
#   end
#   for n1 in 1 : (N - 1)   for n2 in (n1 + 1) : N
#     close = true; r2 = 0.; XX = zeros(D); VV = zeros(D);
#     for i in 1 : D
#       XX[i] = X[n1, i] - X[n2, i];
#       VV[i] = V[n1, i] - V[n2, i];
#       while (XX[i] > pi) XX[i] -= 2. * pi; end
#       while (XX[i] < -pi) XX[i] += 2. * pi; end
#       r2 += XX[i] * XX[i];
#       if (r2 > 4. * h * h) close = false; break; end
#     end
#     if (close)
#       if (XX[1]*VV[1] + XX[2]*VV[2] < 0)
#         μ = h*(XX[1]*VV[1] + XX[2]*VV[2])/(sqrt(r2) + 0.01*h^2)
#         Π = (α*c*μ + beta * μ^2)/((rho[n1] + rho[n2])/2)
#       end
#       if (XX[1]*VV[1] + XX[2]*VV[2] >= 0)
#         Π = 0.0
#       end
#       tmp = m * (Pres(rho[n1])/rho[n1]^2 + Pres(rho[n2])/rho[n2]^2 + Π) * H(sqrt(r2));  #conservative Presure term
#       for i in 1 : D
#         A[n1, i] -= XX[i] * (tmp);
#         A[n2, i] += XX[i] * (tmp);
#       end
#     end
#   end   end
#   return A
# end
#
#
# Total(α, beta) = sum(obtain_A(α, beta))
# display(Total(α, beta))
#
#
# g2 = Zygote.gradient(Total, randn(N,D))
# display(g2)












#------------

#------------

#----------- Plotting the automatic diffs of nohash functions

#
# g2 = Zygote.gradient(W, 1.0)
#
# display(g2[1])
#
# G2(x) = Zygote.gradient(W, x)
#
# M = 200
# x_grid = 2 .* h .* rand(M)
# # g2_data = G2.(x_grid)
# # g2 = convert(Vector, g2_data)
# # println(g2)
# # println((g2_data))
# # println(type(g2_data))
#
# #2 print()
# g2_data = zeros(M)
# g2_data_r = zeros(M)
# for i in 1 : M
#     g2_data[i] = G2(x_grid[i])[1]
#     g2_data_r[i] = G2(x_grid[i])[1]/x_grid[i]
# end
#
#
# scatter(x_grid, W.(x_grid))
# scatter!(x_grid, g2_data)
# scatter!(x_grid, g2_data_r, m = ([:star7], 12))
# scatter!(x_grid, H.(x_grid))
