"""
Simulating sph particles with nohash AV

The goal is to rewrite this simulator to be differentiable

Q: can we write the entire code to have no Mutating arrays.


"""

using Plots
using NPZ

#-----------Parameter study -------
T = 70
c = 10.0
g = 7
h = 0.8
α = -0.5
beta = 1.0
cdt = 0.4


power = 6
D = 2; N = round(Int, 2^power); # dimension and number of particles
X = zeros(N, D); V = zeros(N, D); A = zeros(N, D);
rho = zeros(N); m = (2. * pi)^D / N; # average density = 1 = rho_0
pos, vel= zeros(T+1,N,D), zeros(T+1,N,D)


dt = cdt * h / c;
c0 = c^2/g


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


function obtain_rho()
  global X, rho, h;
  for n in 1 : N   rho[n] = m * W(0.);   end
  for n1 in 1 : (N - 1)   for n2 in (n1 + 1) : N
    close = true; r2 = 0.;
    for i in 1 : D
      XX = X[n1, i] - X[n2, i];
      while (XX > pi)   XX -= 2. * pi;   end
      while (XX < -pi)   XX += 2. * pi;   end
      r2 += XX * XX;
      if (r2 > 4. * h * h)
        close = false; break;
      end
    end
    if (close)
      tmp = m * W(sqrt(r2)); rho[n1] += tmp; rho[n2] += tmp;
    end
  end   end
  return rho
end


function P_d_rho2(rho)
  global g
  return (c0 * (rho^g - 1.))
end


function obtain_A()
  global X, A, rho, h, V;
  μ = 0.0
  Π = 0.0
  for n in 1 : N # four vortices on a 2D torus
    A[n, 1] = sin(X[n, 1]) * cos(X[n, 2]);
    A[n, 2] = -cos(X[n, 1]) * sin(X[n, 2]);
  end
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
      tmp = m * (P_d_rho2(rho[n1])/rho[n1]^2 + P_d_rho2(rho[n2])/rho[n2]^2 + Π) * H(sqrt(r2));  #conservative pressure term
      for i in 1 : D
        A[n1, i] -= XX[i] * (tmp);
        A[n2, i] += XX[i] * (tmp);
      end
    end
  end   end
  return A
end



#initial conditions seeded
for i in 1 : round(Int, 2^(power/2))   for j in 1 : round(Int, 2^(power/2))
  n = round(Int, 2^(power/2)) * (i - 1) + j;
  X[n, 1] = 2. * pi * ((i - 1.) / sqrt(N) + 0.004 * (rand() - 0.5));
  X[n, 2] = 2. * pi * ((j - 1.) / sqrt(N) + 0.004 * (rand() - 0.5));
end   end
for n in 1 : N
  V[n, 1] = V[n, 2] = 0.;
end
X = mod.(X, 2*pi)

pos[1,:,:] = X
vel[1,:,:] = V
t = 0.0



#-------------- Integration
ii = 1
ii2 = 1
for k in 1 : T
  global X, V, A, t, ii, dt, c, ii2;

  rho = obtain_rho();
  A = obtain_A();

  V += 0.5 * dt * A;
  X += dt * V;
  #back to torus
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


#-----------Outputs


file_out_pos = "./pos_N$(N)_T$(T)_C$(c)_g$(g)_h$(h)_alpha$(α)_beta$(beta)_av_wcsph.mp4"


pos_data = "./data/pos_N$(N)_T$(T)_C$(c)_g$(g)_h$(h)_alpha$(α)_beta$(beta)_av_wcsph.npy"
vel_data = "./data/vel_N$(N)_T$(T)_C$(c)_g$(g)_h$(h)_alpha$(α)_beta$(beta)_av_wcsph.npy"


function simulate()
    gr(size=(1300,900))
    println("**************** Simulating the particle flow ***************")
    tskip = 2

    #theme(:juno)
    anim = @animate for i ∈ 1 : tskip : T
         n_2 = round(Int,N/2)
         scatter(pos[i, 1:n_2, 1], pos[i, 1:n_2, 2], title = "differentiable sph engine", xlims = [0, 2*pi], ylims = [0,2*pi], legend = false)
         scatter!(pos[i, (n_2+1):end, 1], pos[i, (n_2+1):end, 2], color = "red")
    end
    gif(anim, file_out_pos, fps = round(Int, T/4))
    println("****************  Simulation COMPLETE  *************")
end

simulate()


npzwrite(pos_data, pos)
npzwrite(vel_data, vel)




# tpl = 1:T
# tpl2 = 0:T
#
# plt_den = plot(tpl2, dens[1:(T+1),:], title = "desity fluctuations , N = 4096", xlabel = "time steps", ylabel = "ρ(t) - ρ₀", legend = false, size = (900,600))
# savefig(plt_den, fig_density)
#
# plt_mv = plot(tpl, max_v[1:T], title = "maximum velocity , N = 4096", xlabel = "time steps", ylabel = "V_max(t)", legend = false, size = (900,600))
# savefig(plt_mv, fig_max_v)
# # png(plt_mv, fig_max_v)
