# # Example: Blowfly Model
# Likelihood-free inference for the blow-fly model was introduced by [Simon N.
# Wood](http://dx.doi.org/10.1038/nature09319). We model here the discrete time
# stochastic dynamics of the size ``N`` of an adult blowfly population as given in [section 1.2.3 of the supplementary
# information](https://static-content.springer.com/esm/art%3A10.1038%2Fnature09319/MediaObjects/41586_2010_BFnature09319_MOESM302_ESM.pdf).
# ```math
# N_{t+1} = P N_{t-\tau}\exp(-N_{t-\tau}/N_0)e_t + N_t\exp(-\delta \epsilon_t)
# ```
# where ``eₜ`` and ``ϵₜ`` are independent Gamma random deviates with
# mean 1 and variance ``σp²`` and ``σd²``, respectively.
using Distributions, StatsBase, LikelihoodfreeInference
Base.@kwdef struct BlowFlyModel
    burnin::Int = 50
    T::Int = 1000
end
function (m::BlowFlyModel)(P, N₀, σd, σp, τ, δ)
    p1 = Gamma(1/σp^2, σp^2)
    p2 = Gamma(1/σd^2, σd^2)
    T = m.T + m.burnin + τ
    N = fill(180., T)
    for t in τ+1:T-1
        N[t+1] = P * N[t-τ] * exp(-N[t-τ]/N₀)*rand(p1) + N[t]*exp(-δ*rand(p2))
    end
    N[end-m.T+1:end]
end

# Let us plot four realizations from this model with the same parameters.
using StatsPlots
gr()
m = BlowFlyModel()
plot([plot(m(29, 260, .6, .3, 7, .2),
           xlabel = "t", ylabel = "N", legend = false) for _ in 1:4]...,
     layout = (2, 2))


# To compare different realizations we will use histogram summary statistics.
# In the literature one finds also other summary statistics for this data.
summary_statistics(N) = fit(Histogram, N, 140:16:16140).weights

# We will use a normal prior on log-transformed parameters.
function parameter(logparams)
    lP, lN₀, lσd, lσp, lτ, lδ = logparams
    (P = round(exp(2 + 2lP)),
    N₀ = round(exp(4 + .5lN₀)),
    σd = exp(-.5 + lσd),
    σp = exp(-.5 + lσp),
    τ = round(Int, max(1, min(500, exp(2 + lτ)))),
    δ = exp(-1 + .4lδ))
end
(m::BlowFlyModel)(logparams) = m(parameter(logparams)...)
target(m::BlowFlyModel) = [(log(29) - 2)/2,
                           (log(260) - 4)*2,
                           log(.6) + .5,
                           log(.3) + .5,
                           log(7) - 2,
                           (log(.2) + 1)/.4]
lower(m::BlowFlyModel) = fill(-5., 6)
upper(m::BlowFlyModel) = fill(5., 6)
prior = TruncatedMultivariateNormal(zeros(6), ones(6),
                                    lower = lower(m), upper = upper(m))

# Let us now generate some target data.
model = BlowFlyModel()
x0 = target(model)
data = summary_statistics(model(x0))

# ## Adaptive SMC
smc = AdaptiveSMC(prior = prior)
result = run!(smc, x -> summary_statistics(model(x)), data,
              maxfevals = 2*10^5, verbose = false)
using PrettyTables
pretty_table([[keys(parameter(zeros(6)))...] quantile(smc, .05) median(smc) mean(smc) x0 quantile(smc, .95)],
             ["names", "5%", "median", "mean", "actual", "95%"],
             formatter = ft_printf("%10.3f"))
#-
histogram(smc)
#-
corrplot(smc)

# ## KernelABC
k = KernelABC(prior = prior, delta = 1e-1, K = 10^3, kernel = Kernel())
result = run!(k, x -> summary_statistics(model(x)), data)
pretty_table([[keys(parameter(zeros(6)))...] quantile(k, .05) median(k) mean(k) x0 quantile(k, .95)],
             ["names", "5%", "median", "mean", "actual", "95%"],
             formatter = ft_printf("%10.3f"))
#-
histogram(k)

# ## Kernel Recursive ABC (with callback)
k = KernelRecursiveABC(prior = prior,
                       K = 100,
                       delta = 1e-3,
                       kernel = Kernel(bandwidth = Bandwidth(heuristic = MedianHeuristic(2^3))),
                       kernelx = Kernel());
# We will use a callback here to show how the estimated parameters evolves.
using LinearAlgebra
res_krabc = run!(k, x -> summary_statistics(model(x)), data,
                 maxfevals = 1300,
                 verbose = true,
                 callback = () -> @show norm(k.theta - x0)/norm(x0))

