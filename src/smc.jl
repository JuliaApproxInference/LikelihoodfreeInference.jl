"""
    PMC(; epsilon, K, prior, ess_min = 0.5, proposal = defaultproposal(prior))

Population Monte Carlo ABC structure, with `epsilon` schedule, `K` particles,
`prior`, resampling threshold `ess_min` and `proposal` distribution.
The `epsilon` schedule should be an iterable, e.g. an array of numbers,
[`EpsilonExponentialDecay`](@ref) or [`EpsilonLinearDecay`](@ref).
One dimensional problems should be defined in vectorized form (
see [`run!`](@ref) for an example).

# Example
```julia-repl
pmc = PMC(epsilon = EpsilonExponentialDecay(1, .01, .9), K = 20,
                 prior = MultivariateUniform([-1, -1], [1, 1]))
model(theta) = .1 * randn(2) .+ theta
data = [.3, -.2]
run!(pmc, model, data)

mean(pmc)
particles(pmc)
weights(pmc)
```

[Beaumont, M. A., Cornuet, J., Marin, J., and Robert, C. P. (2009)
Adaptive approximate Bayesian computation. Biometrika,96, 983-990.
](http://dx.doi.org/10.1093/biomet/asp052)
"""
struct PMC{E,P,D,D1,T} <: AbstractABC
    epsilon::E
    log_ess_min::Float64
    particles::Vector{P}
    log_weights::Vector{Float64}
    prior::D
    proposal::D1
    distance::T
end
function PMC(; epsilon, prior::Distribution{T}, K = 10^3, ess_min = 0.5,
             proposal = defaultproposal(prior), distance = euclidean) where T
    particles = init_particles(prior, K)
    PMC(epsilon, log(ess_min), particles, fill(-log(K), K),
        prior, proposal(particles), distance)
end
function Base.show(io::IO, mime::MIME"text/plain", pmc::PMC)
    println(io, "PMC(K = $(length(pmc.particles)), ess_min = $(exp(pmc.log_ess_min)))")
    print(io, "  epsilon schedule: ")
    show(io, mime, pmc.epsilon)
    print(io, "  prior: ")
    show(io, mime, pmc.prior)
    print(io, "  proposal: ")
    show(io, mime, pmc.proposal)
end
weights(pmc::PMC) = weights(pmc.log_weights)
particles(pmc::PMC) = pmc.particles

sample_particles!(particles, distribution, eps, model, data, distance) =
    sample_particles!(Random.GLOBAL_RNG, particles, distribution, eps, model, data, distance)
function sample_particles!(rng::Random.AbstractRNG,
                           particles, distribution, eps, model, data, distance)
    n_sims = 0
    Threads.@threads for particle in particles
        while true
            rand!(rng, distribution, particle)
            sim_data = model(particle)
            n_sims += 1
            distance(sim_data, data) < eps && break
        end
    end
    n_sims
end
resample(particles, weights; verbose = false, log_weights = false) = resample!(Random.GLOBAL_RNG, copy(particles), weights, verbose = verbose, log_weights = log_weights)[1]
function resample!(rng::Random.AbstractRNG, particles, weights;
                   verbose = true,
                   log_weights = true)
    verbose && println("Resampling...")
    idxs = wsample(1:length(weights), log_weights ? exp.(weights) : weights, length(weights))
    old_particles = deepcopy(particles) # only particles[idxs] would be needed
    for (i, j) in enumerate(idxs)
        copyto!(particles[i], old_particles[j])
    end
    weights .= log_weights ? -log(length(weights)) : 1/length(weights)
    particles, weights
end
function update_log_weights!(log_weights, particles, prior, proposal)
    for i in eachindex(log_weights)
        log_weights[i] = logpdf(prior, particles[i]) - logpdf(proposal, particles[i])
    end
    log_weights .-= logsumexp(log_weights)
    log_weights
end

"""
    run!([rng], method, model, data;
         callback = () -> nothing, maxfevals = Inf, verbose = true)

Run `method` on `model` and `data`.
The function `callback` gets evaluated after every iteration.
The method stops after the first iteration that reaches more than `maxfevals` calls to the model.
# Example
```julia-repl
using Distributions, LinearAlgebra
pmc = PMC(epsilon = [1, .5, .2, .1, .01, .001], K = 10^3,
                 prior = TruncatedMultivariateNormal([1.], 2I,
                                                     lower = [0], upper = [3]))
model(theta) = theta * randn() .+ 1.2
data = [.5]
callback() = @show mean(pmc)
run!(pmc, model, data, callback = callback, maxfevals = 10^5)

using StatsBase, StatsPlots
h = fit(Histogram, vcat(particles(pmc)...), nbins = 50)
StatsPlots.plot(h)
```
"""
run!(pmc, model, data; kwargs...) = run!(Random.GLOBAL_RNG, pmc, model, data; kwargs...)
# Beaumont et al. 2009 (see also Lintusaari et al. 2016
function run!(rng::AbstractRNG, pmc::PMC, model, data;
              callback = () -> nothing, maxfevals = Inf, verbose = true)
    start_time = now()
    n_sims = Int[]
    K = length(pmc.particles)
    push!(n_sims, sample_particles!(rng, pmc.particles, pmc.prior,
                                    first(pmc.epsilon), model, data, pmc.distance))
    update!(pmc.proposal, pmc.particles, pmc.log_weights)
    callback()
    verbose && @printf "%10.s %13.s %8.s %8.s %8.s\n" "iteration" "elapsed" "fevals" "epsilon" "ess"
    verbose && @printf "%10.s %13.s %8.s %8.e %8.s\n" 0 round(now() - start_time, Second) n_sims[1] first(pmc.epsilon) 1
    for (i, eps) in enumerate(Iterators.drop(pmc.epsilon, 1))
        push!(n_sims, sample_particles!(rng, pmc.particles, pmc.proposal, eps, model, data, pmc.distance))
        update_log_weights!(pmc.log_weights, pmc.particles, pmc.prior, pmc.proposal)
        l_ess = log_ess(pmc.log_weights) - log(K)
        if l_ess < pmc.log_ess_min
            resample!(rng, pmc.particles, pmc.log_weights, verbose = verbose)
        end
        update!(pmc.proposal, pmc.particles, pmc.log_weights)
        callback()
        verbose && @printf "%10.s %13.s %8.s %8.e %8.s\n" i round(now() - start_time, Second) sum(n_sims) eps exp(l_ess)
        sum(n_sims) > maxfevals && break
    end
    (weights = exp.(pmc.log_weights),
     particles = pmc.particles,
     n_sims = n_sims)
end

struct Particle{T}
    θ::Vector{Float64}
    x::T
    distances::Vector{Float64}
end
function Base.copyto!(dest::Particle, bc::Particle)
    dest.θ .= bc.θ
    for k in eachindex(dest.x)
        dest.x[k] .= bc.x[k]
    end
    dest.distances .= bc.distances
    dest
end
theta(p::Particle) = p.θ

"""
    AdaptiveSMC(; alpha = 0.9, epsilon = .1, ess_min = 0.5, M = 1,
                  prior, K = 10^3, proposal = defaultproposal(prior))

Adaptive Sequential Monte Carlo structure with `K` particles,
`M` calls of the model per parameter value, final `epsilon`,
decrease parameter `alpha` and resampling threshold `ess_min`.
See also [`PMC`](@ref).

# Example
```
using Distributions, LinearAlgebra
asmc = AdaptiveSMC(prior = MultivariateNormal(zeros(2), 4I))
model(theta) = vcat([.3*randn(2) .+ theta for _ in 1:2]...) # 2 i.i.d samples
data = [.4, -.3, .5, -.2]
run!(asmc, model, data)

mean(asmc)
using StatsPlots
p = hcat(particles(asmc)...); StatsPlots.scatter(p[1, :], p[2, :])
```

[Del Moral, P., Doucet, A., and Jasra, A. (2012) An adaptive
sequential Monte Carlo method for approximate Bayesian
computation. Statistics and Computing, 22, 1009-1020
](http://dx.doi.org/10.1007/s11222-011-9271-y)
"""
struct AdaptiveSMC{P,P1,D,T} <: AbstractABC
    α::Float64
    ϵ::Float64
    Nt::Float64
    prior::P
    proposal::P1
    particles::Vector{Particle{T}}
    weights::Vector{Float64}
    distance::D
end
function AdaptiveSMC(; alpha = 0.9, epsilon = .1, ess_min = 0.5, M = 1,
                     prior, K = 10^3,
                     proposal = defaultproposal(prior),
                     distance = euclidean,
                     datatype = distance == euclidean ? Vector{Float64} :
                                                        Vector{Vector{Float64}}
                    )
    particles = [Particle(randn(eltype(prior), length(prior)),
                          [datatype(undef, 0) for _ in 1:M],
                          zeros(M)) for _ in 1:K]
    AdaptiveSMC(alpha, float(epsilon), float(ess_min * K), prior,
                proposal(theta.(particles)),
                particles, zeros(K), distance)
end
Distributions.mean(x::Union{<:PMC, <:AdaptiveSMC}) = sum(particles(x) .* weights(x))
weights(asmc::AdaptiveSMC) = asmc.weights
particles(asmc::AdaptiveSMC) = theta.(asmc.particles)
function Base.show(io::IO, mime::MIME"text/plain", asmc::AdaptiveSMC)
    println(io, "AdaptiveSMC(α = $(asmc.α), ϵ = $(asmc.ϵ), K = $(length(asmc.particles)), ess_min = $(asmc.Nt/length(asmc.particles)), M = $(length(asmc.particles[1].distances)))")
    print(io, "  prior: ")
    show(io, mime, asmc.prior)
    print(io, "  proposal: ")
    show(io, mime, asmc.proposal)
end
function ess!(weights, particles, factor, e)
    for i in eachindex(weights)
        weights[i] = factor[i] * sum(particles[i].distances .< e)
    end
    maximum(weights) == 0 && return 0
    weights ./= sum(weights)
    ess(weights)
end
function getfactor(weights, particles, e_old)
    factor = zeros(length(weights))
    smallest_dist = Inf
    for i in eachindex(factor)
        weights[i] == 0 && continue
        tmp = minimum(particles[i].distances)
        tmp < smallest_dist && (smallest_dist = tmp)
        denom = sum(particles[i].distances .< e_old)
        factor[i] = weights[i] / denom
    end
    smallest_dist, factor
end
function new_epsilon!(weights, e_old, particles, alpha, e_min)
    smallest_dist, factor = getfactor(weights, particles, e_old)
    target = alpha * ess(weights)
    e = find_zero(e -> target - ess!(weights, particles, factor, e),
                  (smallest_dist, e_old), Bisection())
    e = max(e, e_min)
    e, ess!(weights, particles, factor, e)
end
function sample_particles!(rng::Random.AbstractRNG,
                           particles::Vector{<:Particle},
                           distribution, model, data, weights, distance;
                           indexed = false)
    n_sims = 0
    for (i, (w, particle)) in enumerate(zip(weights, particles))
        w == 0 && continue
        if indexed
            Distributions._rand!(rng, distribution, particle.θ, i)
        else
            rand!(rng, distribution, particle.θ)
        end
        for k in eachindex(particle.x)
            particle.x[k] = model(particle.θ)
            n_sims += 1
            particle.distances[k] = distance(particle.x[k], data)
        end
    end
    n_sims
end
function mh_accept!(particles, old_particles, prior, ϵ, weights)
    for i in eachindex(particles)
        weights[i] == 0 && continue
        logw = logpdf(prior, particles[i].θ) - logpdf(prior, old_particles[i].θ) +
               log(mean(particles[i].distances .< ϵ)/mean(old_particles[i].distances .< ϵ))
        if logw >= 0 || rand() < exp(logw)
            continue
        end
        copyto!(particles[i], old_particles[i]) # proposal rejected
    end
    particles
end

function run!(rng::Random.AbstractRNG, asmc::AdaptiveSMC, model, data;
              callback = () -> nothing,
              maxfevals = Inf,
              verbose = true)
    start_time = now()
    n_sims = Int[]
    epsilons = Float64[]
    i = 0
    eps = Inf
    K = length(asmc.particles)
    asmc.weights .= fill(1/K, K)
    push!(n_sims, sample_particles!(rng, asmc.particles, asmc.prior, model, data, asmc.weights, asmc.distance))
    update!(asmc.proposal, theta.(asmc.particles), asmc.weights, log_weights = false)
    callback()
    verbose && @printf "%10.s %13.s %8.s %8.s %8.s\n" "iteration" "elapsed" "fevals" "epsilon" "ess"
    verbose && @printf "%10.s %13.s %8.s %8.e %8.s\n" 0 round(now() - start_time, Second) n_sims[1] eps 1
    while true
        i += 1
        eps, ess = new_epsilon!(asmc.weights, eps, asmc.particles, asmc.α, asmc.ϵ)
        push!(epsilons, eps)
        ess < asmc.Nt && resample!(rng, asmc.particles, asmc.weights,
                                  log_weights = false, verbose = verbose)
        old_particles = deepcopy(asmc.particles)
        update!(asmc.proposal, theta.(asmc.particles), asmc.weights, log_weights = false)
        push!(n_sims, sample_particles!(rng, asmc.particles, asmc.proposal, model, data, asmc.weights, asmc.distance, indexed = true))
        mh_accept!(asmc.particles, old_particles, asmc.prior, eps, asmc.weights)
        ess = ess!(asmc.weights, asmc.particles, ones(K), eps)
        callback()
        verbose && @printf "%10.s %13.s %8.s %8.e %8.s\n" i round(now() - start_time, Second) sum(n_sims) eps ess/K
        (eps ≈ asmc.ϵ || sum(n_sims) > maxfevals) && break
    end
    (weights = asmc.weights,
     particles = asmc.particles,
     n_sims = n_sims,
     epsilons = epsilons)
end
