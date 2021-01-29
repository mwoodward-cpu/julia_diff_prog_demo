"""
Differentiable md solver (U motivated by SPH) equation with diffusion term added
Method :: MD_diff
Flow   :: Taylor-green vortex decay (simple no external forcing)
"""


using Zygote
using Plots



h = 0.25
power = 10
D = 2; N = round(Int, 2^power); # dimension and number of particles
X = zeros(N, D); V = zeros(N, D); A = zeros(N, D);
As = zeros(D);
U0 = 2.0
lambda = 1e-2

"""
FORWARD SIMULATOR FUNCTIONS
"""
function U(r::Float64)
      q = r/h
      if (q >= 2.0)
            return 0.0
      end
      if (q >= 1.0)
            return U0 *(2.0 - q)^(3.0)/4.0
      end
      return U0 * (1.0 - 1.5 * q^2 * (1.0 - q/2.0))
end


function dU_dr_r(r::Float64)
      q = r/h
      if (q >= 2.0)
            return 0.0
      end
      if (q >= 1.0)
            return -3.0*U0*(2.0 - q)^2/(4.0*r*h)
      end
      return U0*(-3.0 + 9.0*q/4.0)/(h^2)
end



n_hash = floor(Int, 2. * pi / h);   l_hash = 2. * pi / n_hash;
function obtain_MD_diff_A()
  global A, X, V;
  A = zeros(N,D);
  VV = zeros(D);
  XX = zeros(D);
# putting coordinates inside the (2 pi)^2 torus, building hash
  hash = [Set() for i in 1 : n_hash, j in 1 : n_hash];
  for n in 1 : N
    for i in 1 : D
      while (X[n, i] < 0.)   X[n, i] += 2. * pi;   end
      while (X[n, i] > 2. * pi)   X[n, i] -= 2. * pi;  end
    end
    push!(hash[floor(Int, X[n, 1] / l_hash) + 1,
               floor(Int, X[n, 2] / l_hash) + 1], n);
  end

  # XX = zeros(D);
  # VV = zeros(D);
  for n in 1 : N
    # A[n, :] = obtain_forcing_A(X[n, :]);
    x_hash = [floor(Int, X[n, 1] / l_hash) + 1,
              floor(Int, X[n, 2] / l_hash) + 1];
    for xa_hash in x_hash[1] - 2 : x_hash[1] + 2
      xb_hash = xa_hash;    while (xb_hash < 1)    xb_hash += n_hash;   end
      while (xb_hash > n_hash)    xb_hash -= n_hash;   end
      for ya_hash in x_hash[2] - 2 : x_hash[2] + 2
        yb_hash = ya_hash;    while (yb_hash < 1)    yb_hash += n_hash;   end
        while (yb_hash > n_hash)    yb_hash -= n_hash;   end
        for n2 in hash[xb_hash, yb_hash]
          close = true;   r2 = 0.;
          for i in 1 : D
            XX[i] = X[n, i] - X[n2, i];
            VV[i] = V[n, i] - V[n2, i];
            while (XX[i] > pi)   XX[i] -= 2. * pi;   end
            while (XX[i] < -pi)   XX[i] += 2. * pi;   end
            r2 += XX[i] * XX[i];
            if (r2 > 4. * h * h)   close = false; break;   end
          end
          if (close)
            tmp_potl = - dU_dr_r(sqrt(r2));     #potential force ampl
            tmp_diff = - lambda * U(sqrt(r2))   #diffusion force ampl
            for i in 1 : D   A[n,i] += tmp_potl * XX[i] + tmp_diff * VV[i];   end
          end
        end
    end   end
  end
end




"""
DIFFABLE SIMULATOR FUNCTIONS WITH TESTING
"""

# function obtain_MD_diff_A_scaler(X, V, U0, lam, i)
function obtain_MD_diff_A_scaler(X, i)
  """
  Computes the acceleration due to gradient of potential (pairwise, inspired by sph)
  of particle i with pairwise interaction over j.
  A physical guess for diffusion is added
  Currentely this is without neighborlists (no hash)

  Need to find a way to avoid mutating arrays XX and VV
  """
  global As, X, V
  As = zeros(D);
  XX = zeros(D);
  VV = zeros(D);

  for j ∈ 1 : N
    r2 = 0.0
    # for k in 1 : D
      XX = X[i, :] - X[j, :];
      VV = V[i, :] - V[j, :];
      XX = minimum.(abs.(XX), 2*pi .- abs.(XX))
      # if (XX[k] > pi)
      #   XX[k] -= 2. * pi;
      # end
      # if (XX[k] < -pi)
      #   XX[k] += 2. * pi;
      # end
      # r2 += XX^2;
    # end
    r2 = sum(XX.^2);

    if (r2 <= 4.0 * h^2)
      tmp_pot = - dU_dr_r(sqrt(r2));
      tmp_dff = -lambda * U(sqrt(r2));
      As .+= tmp_pot .* (XX) .+ tmp_dff .* (VV)
    end
  end
  return As
end


# function obtain_vec_A(X, V, U0, lam)
function obtain_vec_A(X)
  """
  From obtain_scaler, this computes the vector A over all N particles
  """
  I = 1 : (N)   #indicies over i ∈ {1, 2, ... N}
  # A_ = obtain_MD_diff_A_scaler.(Ref(X, V, U0, lam), I)
  A_ = obtain_MD_diff_A_scaler.(Ref(X), I)
  return A_
end


ii = 1
function verlet(X, V, T, c, α, beta)
  """
  computes the trajectories and velocities of lagrangian particles
  solved using SPH
  Need to avoid mutating array in last few lines
  """
  for k in 1 : T
    global ii, dt, c

    A = obtain_MD_diff_A();

    V += 0.5 * dt * A;
    X += dt * V;
    X = mod.(X, 2*pi)

    A = obtain_MD_diff_A();

    V += 0.5 * dt * A;
    t += dt;

    vel[k + 1, :, :] = V
    pos[k + 1, :, :] = X
    println("time step:", ii)
    ii += 1
  end
  return pos, vel
end



#initial conditions seeded
for i in 1 : round(Int, 2^(power/2))   for j in 1 : round(Int, 2^(power/2))
  n = round(Int, 2^(power/2)) * (i - 1) + j;
  X[n, 1] = 2. * pi * ((i - 1.) / sqrt(N) + 0.0001 * (rand() - 0.5));
  X[n, 2] = 2. * pi * ((j - 1.) / sqrt(N) + 0.0001 * (rand() - 0.5));
end   end
for n in 1 : N
  V[n, 1] = sin(X[n, 1]) * cos(X[n, 2]) #* cos(X[n, 3])
  V[n, 2] = -cos(X[n, 1]) * sin(X[n, 2]) #* cos(X[n, 3]);
end
X = mod.(X, 2*pi)


#--------------- Testing functions
"""
Here we test the ability of AD on our individual functions
  NOTES:
        > U(r) is compatable with zygote.gradient and matches dU_dr
        > Currently As is not matching the true A.

"""

"""
> U(r) is compatable with zygote.gradient and matches dU_dr
"""

# #U and dU_dr_r
# G2(x) = Zygote.gradient(U, x)
# M = 200
# x_grid = 2 .* h .* rand(M)
# g2_data = zeros(M)
# g2_data_r = zeros(M)
# for i in 1 : M
#     g2_data[i] = G2(x_grid[i])[1]
#     g2_data_r[i] = G2(x_grid[i])[1]/x_grid[i]
# end
#
#
# scatter(x_grid, U.(x_grid))
# scatter!(x_grid, g2_data)
# scatter!(x_grid, g2_data_r, m = ([:star7], 12))
# scatter!(x_grid, dU_dr_r.(x_grid))





obtain_MD_diff_A()
# A_vector = obtain_vec_A(X)
display(A)
# display(size(A_vector))
# display(A .- A_vector)
Asc = obtain_MD_diff_A_scaler(X, 1)
print(Asc)

t_sum(x) = sum(obtain_MD_diff_A_scaler(x, 1))
g_test = Zygote.gradient(t_sum, X)
display(g_test)

# t_sum(X) = sum(obtain_vec_rho(X))
# t_sum(X) = sum(abs2, obtain_vec_rho(X))
# g4 = Zygote.gradient(t_sum, X)
# println("vec_rho grad = ", g4)


# t3_sum(X) = sum(obtain_rho_scaler.(Ref(X), I))
# t3_sum(X) = sum(obtain_vec_rho(X))
# g3s = Zygote.gradient(t2_sum, X)
# println(g3s[1])
