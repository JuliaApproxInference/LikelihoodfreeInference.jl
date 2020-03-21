"""
    MultivariateUniform(lower, upper)

Uniform distribution in `d` dimensions, where `d == length(lower) == length(upper)`.
# Example
```julia-repl
julia> d = MultivariateUniform([-2., -3], [4, 1])
MultivariateUniform{Float64} in 2 dimensions

julia> logpdf(d, [-3, 0])
-Inf

julia> logpdf(d, [-1, 0])
-3.1780538303479453
```
"""
struct MultivariateUniform{S} <: ContinuousMultivariateDistribution
    lower::Vector{S}
    upper::Vector{S}
    range::Vector{S}
    function MultivariateUniform(lower, upper)
        length(lower) != length(upper) && error("Lower and upper bounds must have the same length.")
        (|)((lower .> upper)...) && error("Upper bounds must not be smaller than lower bounds")
        new{eltype(lower)}(lower, upper, upper .- lower)
    end
end
Base.length(d::MultivariateUniform) = length(d.lower)
lower(d::MultivariateUniform) = d.lower
upper(d::MultivariateUniform) = d.upper
function Base.show(io::IO, ::MIME"text/plain", d::MultivariateUniform{S}) where S
    println(io, "MultivariateUniform{$S} in $(length(d.lower)) dimensions")
end
function Distributions._rand!(rng::Random.AbstractRNG,
                              d::MultivariateUniform,
                              x::AbstractVector{<:Real})
#     length(x) != length(d) && error("length(x) = $(length(x)), but length(distribution) = $(length(d))")
    for i in eachindex(d.lower)
        x[i] = d.lower[i] + rand(rng) * d.range[i]
    end
    x
end
function Distributions._logpdf(d::MultivariateUniform, x::AbstractVector{<:Real})
    for i in eachindex(x)
        (d.lower[i] > x[i] || d.upper[i] < x[i]) && return -Inf
    end
    -sum(log.(d.range))
end
Distributions.logpdf(d::Nothing, ::Any) = 0.

"""
    TruncatedMultivariateNormal(mvnormal, lower, upper)

Multivariate normal distribution `mvnormal` truncated to a box
given by `lower` and `upper` bounds.
Simple rejection sampling is implemented.
IMPORTANT: `logpdf` is not properly normalized.
# Example
```julia-repl
julia> using Distributions

julia> d = TruncatedMultivariateNormal(MultivariateNormal([2., 3.], .3*I), [0, 0], [4, 5]);

julia> logpdf(d, [2, 3])
-0.6339042620834094

julia> logpdf(d, [2, 3]) == logpdf(d.mvnormal, [2, 3])
true

julia> logpdf(d, [-1, 1])
-Inf
```
"""
struct TruncatedMultivariateNormal{B, T} <: ContinuousMultivariateDistribution
    mvnormal::B
    lower::Vector{T}
    upper::Vector{T}
end
"""
    TruncatedMultivariateNormal(m, cov; lower, upper)

Constructs a `TruncatedMultivariateNormal` with mean `m` covariance matrix `cov`
and bounds `lower` and `upper`.
# Example
```julia-repl
julia> d = TruncatedMultivariateNormal([0, 0], 3I;
                                       lower = [-1, -4], upper = [4, 5])
```
"""
function TruncatedMultivariateNormal(m, sig; lower, upper)
    TruncatedMultivariateNormal(MultivariateNormal(m, sig), lower, upper)
end
Base.length(d::TruncatedMultivariateNormal) = length(d.lower)
lower(d::TruncatedMultivariateNormal) = d.lower
upper(d::TruncatedMultivariateNormal) = d.upper
function Distributions._rand!(rng::Random.AbstractRNG,
                              d::TruncatedMultivariateNormal,
                              x::AbstractVector{<:Real})
    # To do this more efficiently
    # see e.g. https://github.com/BrianNaughton/TruncatedMVN.jl/blob/master/src/multivariate.jl
    while true
        Distributions._rand!(rng, d.mvnormal, x)
        for i in eachindex(x)
            (d.lower[i] > x[i] || d.upper[i] < x[i]) && break
            if i == length(x)
                return x
            end
        end
    end
end
function Distributions._logpdf(d::TruncatedMultivariateNormal,
                               x::AbstractVector{<:Real})
    # This is wrong; but it doesn't matter here.
    # A constant is missing due to truncation.
    for i in eachindex(x)
        (d.lower[i] > x[i] || d.upper[i] < x[i]) && return -Inf
    end
    Distributions._logpdf(d.mvnormal, x)
end

"""
    MultiParticleNormal(particles, weights;
                        lower = nothing, upper = nothing,
                        diagonal = false,
                        scaling = 2, regularization = 0.)

Mixture of `k` multivariate Gaussians with means given by an array of `particles`
(`length(particles) == k`), weighted by `weights` and with covariance matrix
given by `scaling * cov(particles) + regularization * I` for each Gaussian,
if `diagonal == false`; otherwise `scaling * std(particles) + regularization`.
Sampling truncated Gaussians with diagonal covariance matrix can be more efficient.
The mean of the distribution is the weighted average of the particles.

If `lower` and `upper` are not `nothing` the Gaussians are truncated to the box
confined by the `lower` and `upper` bounds.
Note that the logpdf of the truncated version is not properly normalized
(see also [`TruncatedMultivariateNormal`](@ref)).
# Example
```julia-repl
julia> d = MultiParticleNormal([[0, 0], [2, 3], [4, 0]], [1/3, 1/3, 1/3]);

julia> mean(d)
2-element Array{Float64,1}:
 2.0
 1.0
```
"""
mutable struct MultiParticleNormal{P, S, B} <: ContinuousMultivariateDistribution
    particles::Vector{P}
    weights::Vector{Float64}
    cov::S
    scaling::Float64
    regularization::Float64
    lower::B
    upper::B
end
"""
    MultiParticleNormal(; kwargs...)

Returns a anonymous function that accepts `particles` as input and creates a
`MultiParticleNormal` distributions with uniform weights. It accepts the same
keyword arguments `kwargs...` as the normal `MultiParticleNormal` constructor.
# Example
```julia-repl
julia> constructor = MultiParticleNormal(scaling = 3, lower = [-1, 0], upper = [5, 5]);

julia> d = constructor([[0, 0], [2, 3], [4, 0], [5, 1]])
MultiParticleNormal(4 particles in 2 dimensions, scaling = 3.0, regularization = 0.0, with bounds)

julia> d.weights
4-element Array{Float64,1}:
 0.25
 0.25
 0.25
 0.25
```
"""
function MultiParticleNormal(; kwargs...)
    particles -> MultiParticleNormal(particles,
                                     fill(1/length(particles), length(particles));
                                     kwargs...)
end
function cov_matrix(::Type{<:PDMats.AbstractPDMat}, particles, weights, scaling, regularization)
    m = scaling .* cov(particles[weights .> 0]) + regularization * I
    for _ in 1:10
        try
            return PDMats.PDMat(m)
        catch err
            if typeof(err)!=LinearAlgebra.PosDefException
                throw(err)
            end
            n = size(m, 1)
            ϵ = 1e-6 * tr(m)/n
            @inbounds for i in 1:n
                m[i, i] += ϵ
            end
        end
    end
end
function cov_matrix(::Type{<:AbstractVector}, particles, weights, scaling, regularization)
    scaling .* std(particles[weights .> 0]) .+ regularization
end
function MultiParticleNormal(particles, weights;
                             lower = nothing, upper = nothing,
                             diagonal = false,
                             scaling = 2, regularization = 0.)
    sum(weights) ≈ 1 || error("Particle weights must be normalized.")
    MultiParticleNormal(particles, weights,
                        cov_matrix(diagonal ? Vector : PDMats.PDMat,
                                   particles, weights, scaling, regularization),
                        float(scaling), float(regularization),
                        lower, upper)
end
Base.length(d::MultiParticleNormal) = size(d.cov, 1)
lower(d::MultiParticleNormal{<:Any, <:Any, <:AbstractVector}) = d.lower
upper(d::MultiParticleNormal{<:Any, <:Any, <:AbstractVector}) = d.upper
function Base.show(io::IO, ::MIME"text/plain", d::MultiParticleNormal{P,S,B}) where {P, S, B}
    println(io, "MultiParticleNormal($(length(d.particles)) particles in $(size(d.cov, 1)) dimensions, scaling = $(round(d.scaling, sigdigits = 3)), regularization = $(round(d.regularization, sigdigits = 3)), $(B <: Nothing ? "without" : "with") bounds)")
end
function Distributions._rand!(rng::Random.AbstractRNG,
                              d::MultiParticleNormal,
                              x::AbstractVector{<:Real})
    idx = wsample(d.weights)
    Distributions._rand!(rng, d, x, idx)
end
function Distributions._rand!(rng::Random.AbstractRNG,
                              d::MultiParticleNormal{<:Any, <:Any, <:Nothing},
                              x::AbstractVector{<:Real},
                              idx::Int)
    Distributions.add!(unwhiten!(d.cov, randn!(rng, x)), d.particles[idx])
end
function Distributions._rand!(rng::Random.AbstractRNG,
                              d::MultiParticleNormal,
                              x::AbstractVector{<:Real},
                              idx::Int)
    while true
        Distributions.add!(unwhiten!(d.cov, randn!(rng, x)), d.particles[idx])
        for i in eachindex(x)
            (d.lower[i] > x[i] || d.upper[i] < x[i]) && break
            if i == length(x)
                return x
            end
        end
    end
        # To do this more efficiently
        # see e.g. https://github.com/BrianNaughton/TruncatedMVN.jl/blob/master/src/multivariate.jl
        # return gibbsTMVN(d.particles[idx], d.cov, d.lower, d.upper, diagm(ones(length(d.upper))), 1)
end
function Distributions._rand!(rng::Random.AbstractRNG,
                              d::MultiParticleNormal{<:Any, <:AbstractVector},
                              x::AbstractVector{<:Real},
                              idx::Int)
    particle = d.particles[idx]
    for i in eachindex(x)
        a = d.lower[i]
        b = d.upper[i]
        r = b - a
        m = particle[i]
        sig = d.cov[i]
        if r/sig <= 2.5066282746310002
            denom = 1/(2*sig^2)
            while true
                tmp = rand() * r + a
                if exp(-(tmp-m)^2*denom) > rand()
                    x[i] = tmp
                    break
                end
            end
        else
            while true
                tmp = m + randn() * sig
                if a <= tmp && b >= tmp
                    x[i] = tmp
                    break
                end
            end
        end
    end
    x
end
function _lpdf(d, x)
    log(sum(d.weights[i] * exp(logpdf(MultivariateNormal(d.particles[i], d.cov), x))
            for i in eachindex(d.weights)))
end
function Distributions._logpdf(d::MultiParticleNormal{<:Any, <:Any, <:Nothing},
                               x::AbstractVector{<:Real})
    _lpdf(d, x)
end
function Distributions._logpdf(d::MultiParticleNormal, x::AbstractVector{<:Real})
    # This is wrong if upper and lower are finite; but it doesn't matter here.
    # A constant is missing due to truncation.
    for i in eachindex(x)
        (d.lower[i] > x[i] || d.upper[i] < x[i]) && return -Inf
    end
    _lpdf(d, x)
end
function Distributions.mean(d::MultiParticleNormal)
    sum(d.weights[i] .* d.particles[i] for i in eachindex(d.weights))
end
function update!(d::MultiParticleNormal{<:Any, S}, particles, weights; log_weights = true) where S
    log_weights ? weights!(d.weights, weights) : d.weights .= weights
    for i in eachindex(particles)
        d.particles[i] .= particles[i]
    end
    d.cov = cov_matrix(S, d.particles, d.weights, d.scaling, d.regularization)
    d
end

lower(d) = fill(-Inf, length(d))
upper(d) = fill(Inf, length(d))
