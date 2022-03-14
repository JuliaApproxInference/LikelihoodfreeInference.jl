# # Example: Gaussian with Given Variance
#
# As a model we are given a univariate Gaussian distribution with unknown mean
# and standard deviation σ = 1. We have one data point at 2.0. For this
# toy example, we can compute the posterior over the mean. But to illustrate
# likelihood-free inference, let us assume here that we can only sample from the
# model:
using LikelihoodfreeInference, Distributions, Random
model(x) = randn() .+ x

# LikelihoodfreeInference.jl passes parameter values as vectors to the model,
# even in the one-dimensional case. In our definition of the model we assume
# that `x[1]` is the mean.
#
# ## Approximate the Posterior
# Our first goal is to find the posterior over the mean given observation and
# a Gaussian prior with mean 0 and standard deviation 5.
data = [2.0]
prior = MultivariateNormal([0.], [5.])

# The true posterior is a Gaussian distribution with mean 25/26*2 and standard
# deviation 26/25
trueposterior = pdf.(Normal.(-1:.01:5, 26/25), 25/26*2.0)
using Plots
gr()
figure = plot(-1:.01:5, trueposterior, label = "posterior")

# Now, we will use an adaptive sequential Monte Carlo method:
smc = AdaptiveSMC(prior = prior, K = 10^4)
result = run!(smc, model, data, verbose = true, maxfevals = 10^6);

# As a Monte Carlo Method the result is a list of particles
particles(smc)
# with corresponding weights
weights(smc)

# The mean of the posterior is given by `weights(smc) .* particles(smc)`, which
# is computed by the `mean` function.
mean(smc)

figure = histogram(vcat(particles(smc)...), weights = weights(smc), normalize = true, label = "AdaptiveSMC")
plot!(figure, -1:.01:5, trueposterior, label = "posterior")

# The `result` above also contains these weights and particles and some
# additional information.
keys(result)

# AdaptiveSMC reduced the epsilon parameter adaptively, as we saw in column
# epsilon of the run above. We can plot this sequence.
scatter(cumsum(result.n_sims)[2:end], result.epsilons,
        yscale = :log10, ylabel = "epsilon", xlabel = "number of model evaluations")

# Alternatively, we may want to use KernelABC.
kabc = KernelABC(prior = prior,
                 kernel = Kernel(),
                 delta = 1e-12,
                 K = 10^4)
result = run!(kabc, model, data, maxfevals = 10^4)
mean(kabc)

figure = histogram(vcat(particles(kabc)...), weights = weights(kabc),
                   xlims = (-1, 5), bins = 100,
                   normalize = true, label = "KernelABC")
plot!(figure, -1:.01:5, trueposterior, label = "posterior")

# ## Point Estimates
# Sometimes we just want a point estimate. We will use BayesianOptimization.jl
# here to minimize the `QDLoss`. We know that the true maximum
# likelihood estimate is at mean = 25/26*2 ≈ 1.923
using BayesianOptimization
p = PointEstimator(optimizer = bo([-10.], [10.]), losstype = QDLoss, prior = prior, K = 100)
result = run!(p, model, data, maxfevals = 5*10^4, verbose = false);
result.x

# KernelRecursiveABC is an alternative method that requires often only few model
# evaluations in low and medium dimensional problems
k = KernelRecursiveABC(prior = prior,
                       kernel = Kernel(),
                       kernelx = Kernel(),
                       delta = 1e-2,
                       K = 100)
result = run!(k, model, data, maxfevals = 2*10^3)
result.x

# ## iid Samples
# Let us suppose here that the data consists of multiple independent and
# identically distributed samples.
data_iid = [[2.0], [1.9], [2.8], [2.1]]

# There are two ways to deal with this data. Either we just assume it is one
# four-dimensional vector
data_onevec = vcat(data_iid...)
# and we define the model as
model_iid_onevec(x) = vcat([model(x) for _ in 1:4]...)
smc = AdaptiveSMC(prior = prior, K = 10^4)
result = run!(smc, model_iid_onevec, data_onevec, verbose = true, maxfevals = 10^6);
histogram(vcat(particles(smc)...), weights = weights(smc), normalize = true, label = "AdaptiveSMC")

# Alternatively, we use another distance function:
model_iid(x) = [model(x) for _ in 1:4]
smc = AdaptiveSMC(prior = prior, K = 10^4, distance = energydistance)
result = run!(smc, model_iid, data_iid, verbose = true, maxfevals = 10^6);
histogram(vcat(particles(smc)...), weights = weights(smc), normalize = true, label = "AdaptiveSMC")
