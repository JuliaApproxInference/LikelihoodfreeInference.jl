"""
Package for likelihood-free inference. Includes Bayesian methods `PMC`, `AdaptiveSMC`, `KernelABC` and `K2ABC`, and different point estimators.
"""
module LikelihoodfreeInference
using Distributions, Random, LinearAlgebra, StatsBase, Dates, PDMats, Roots, Requires, NLopt, Printf, DiffResults, ForwardDiff

export PMC, run!, EpsilonExponentialDecay, EpsilonLinearDecay, MultivariateUniform, AdaptiveSMC, MultiParticleNormal, TruncatedMultivariateNormal, defaultproposal, weights, particles, mean, PointEstimator, QDLoss, KernelLoss, K2ABC, Kernel, Gaussian, ModifiedGaussian, MedianHeuristic, ScottsHeuristic, Bandwidth, Smoothed, Scheduled, StandardMMD, KernelABC, KernelRecursiveABC, LogGaussianKernel, euclidean, energydistance, EnergyLoss, MMDLoss, KLLoss, kldistance, mmd

include("distributions.jl")
include("utils.jl")
include("kernels.jl")
include("smc.jl")
include("pointestimator.jl")
include("kernelabc.jl")
include("krabc.jl")
include("optimization.jl")

function __init__()
    include(joinpath(@__DIR__, "optional_optimization.jl"))
    @require StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd" begin
        function StatsPlots.corrplot(x::AbstractABC;
                                     N = nothing, kwargs...)
            p = particles(x)
            w = weights(x)
            N = N === nothing ? min(1000, 10*length(p)) : N
            M = vcat([hcat(resample(p, w)...)' for _ in 1:div(N, length(p))]...)
            StatsPlots.corrplot(M; kwargs...)
        end
        function StatsPlots.histogram(x::AbstractABC;
                                      layout = nothing,
                                      kwargs...)
            d = length(particles(x)[1])
            n = ceil(Int, sqrt(d))
            layout = layout === nothing ? (n, ceil(Int,d/n)) : layout
            StatsPlots.histogram(hcat(particles(x)...)', weights = weights(x);
                                 layout = layout, legend = false, kwargs...)

        end
    end
end

end # module
