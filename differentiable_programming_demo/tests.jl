#get XX nxn
using Distances

N = 8
D = 2

X = 1:N*D
X = reshape(X, (N,D))
display(X)
r2 =0.0
# for i in 1:N
#     for j in 1:N
#
#         XX1 = X[i,1] - X[j, 1]
#         XX2 = X[i,2] - X[j, 2]
#         r2 = sqrt(XX1^2 + XX2^2)
#     end
# end
#
# print(r2)


# sqrt(sum((x - y) .^ 2))
# print(r)
dist = Euclidean()
R = pairwise(dist, X, X, dims=1)
display(mod.(R, 2*pi))

sigma = (10. / (7. * pi * h * h));
function W(r)
  global h, sigma;
  q = r / h; if (q > 2.)   return 0.;   end
  if (q > 1.)   return (sigma * (2. - q)^3 / 4.);   end
  return (sigma * (1. - 1.5 * q * q * (1. - q / 2.)));
end

Wmat = W.(R)

display(Wmat)
display(sigma)
# function test(r)
#     (r - 3)^2
# end
#
# tbone = test.(R)
# println(tbone)

Y = [0 0; 3 4]
display(Y)

Z = pairwise(dist, Y, Y, dims=1)
display(Z)

Wnew = [0 1; 2 7]

snew = sum(Wnew, dims=2)
display(Wnew)
display(snew)
