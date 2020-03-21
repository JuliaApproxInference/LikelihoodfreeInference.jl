abstract type AbstractABC end
abstract type AbstractPointABC end
weights(x::AbstractABC) = x.weights
particles(x::AbstractABC) = x.particles
Distributions.mean(x::AbstractABC) = sum(weights(x) .* particles(x))
function Distributions.quantile(x::AbstractABC, q)
    [quantile([p[i] for p in particles(x)], Weights(weights(x)), q)
     for i in eachindex(first(particles(x)))]
end
Distributions.median(x::AbstractABC) = quantile(x, .5)

weights(log_weights) = weights!(similar(log_weights), log_weights)
weights!(weights, log_weights) = @. weights = exp(log_weights)

function logsumexp(x::Vector{T}) where T
    res = zero(T)
    for i in eachindex(x)
        res += exp(x[i])
    end
    log(res)
end

log_ess(logw) = -logsumexp(2 .* logw)
ess(w) = 1/sum(w.^2)

euclidean(x, y) = √sum(abs2, x - y)
euclidean(x, y, b::Number) = euclidean(x, y)/b
euclidean(x, y, b::AbstractVector) = √sum(abs2, (x - y) ./ b)
pairwise_euclidean(x::AbstractVector{<:AbstractVector{<:AbstractVector}}) = pairwise_euclidean(vcat(x...))
function pairwise_euclidean(x, b = 1)
    n = length(x)
    result = zeros(n * (n - 1) ÷ 2)
    k = 0
    for i in 1:n
        for j in i+1:n
            k += 1
            result[k] = euclidean(x[i], x[j], b)
        end
    end
    result
end
function energydistance(x, y, b = 1)
    c1 = 0.; c2 = 0.; c3 = 0.
    nx = length(x)
    ny = length(y)
    @simd for i in 1:nx
        for j in 1:ny
            @inbounds c1 += euclidean(x[i], y[j], b)
        end
        for j in i+1:nx
            @inbounds c2 += euclidean(x[i], x[j], b)
        end
    end
    @simd for i in 1:ny
        for j in i+1:ny
            @inbounds c3 += euclidean(y[i], y[j], b)
        end
    end
    2*(c1/(nx*ny) - c2/nx^2 - c3/ny^2)
end



"""
    defaultproposal(prior)

Returns a function that takes a list of `particles` and returns a proposal
distribution. This function is called during initialization of
[`PMC`](@ref) and [`AdaptiveSMC`](@ref).
"""
defaultproposal(d::Any) = particles -> MultiParticleNormal(deepcopy(particles),
                                                           fill(1/length(particles),
                                                           length(particles)))
function defaultproposal(d::Union{<:TruncatedMultivariateNormal, <:MultivariateUniform})
    particles -> MultiParticleNormal(deepcopy(particles),
                                     fill(1/length(particles), length(particles)),
                                     diagonal = length(d.lower) > 10,
                                     lower = d.lower, upper = d.upper)
end
defaultproposal(d::Uniform) = particles -> MultiParticleNormal(deepcopy(particles),
                                                               fill(1/length(particles),
                                                                    length(particles)),
                                                               lower = [d.a],
                                                               upper = [d.b])

init_particles(prior, K) = [randn(eltype(prior), length(prior)) for _ in 1:K]

"""
    EpsilonExponentialDecay(init, last, decay)

Exponentially decreasing sequence starting at `init`, decaying with factor `decay`,
ending as soon as it equals (or drops below) `last`.
# Example
```julia-repl
julia> collect(EpsilonExponentialDecay(1, .2, .5))
4-element Array{Any,1}:
 1.0
 0.5
 0.25
 0.125
```
"""
struct EpsilonExponentialDecay
    init::Float64
    last::Float64
    decay::Float64
end
Base.length(e::EpsilonExponentialDecay) = ceil(Int, -(log(e.init) - log(e.last))/log(e.decay)) + 1
function Base.show(io::IO, ::MIME"text/plain", d::EpsilonExponentialDecay)
    println(io, "EpsilonExponentialDecay(init = $(d.init), last = $(d.last), decay factor = $(d.decay))")
end
function Base.iterate(e::EpsilonExponentialDecay, s = e.init)
    s/e.decay < e.last + 2eps() && return nothing
    (s, s * e.decay)
end

"""
    EpsilonLinearDecay(init, last, decay)

Linearly decreasing sequence starting at `init`, decaying with step size `decay`,
ending as soon as it equals (or drops below) `last`.
# Example
```julia-repl
julia> collect(EpsilonLinearDecay(10, 5, 2))
4-element Array{Any,1}:
 10.0
  8.0
  6.0
  4.0
```
"""
struct EpsilonLinearDecay
    init::Float64
    last::Float64
    decay::Float64
end
Base.length(e::EpsilonLinearDecay) = ceil(Int, (e.init - e.last)/e.decay) + 1
function Base.show(io::IO, ::MIME"text/plain", d::EpsilonLinearDecay)
    println(io, "EpsilonLinearDecay(init = $(d.init), last = $(d.last), delta decay = $(d.decay))")
end
function Base.iterate(e::EpsilonLinearDecay, s = e.init)
    s + e.decay <= e.last && return nothing
    (s, s - e.decay)
end

