struct K2ABC{MMD,K,P,Pa} <: AbstractABC
    kernel::K
    epsilon::Float64
    prior::P
    K::Int
    weights::Vector{Float64}
    particles::Pa
    function K2ABC{MMD}(; kernel, epsilon, prior, K) where MMD
        particles = init_particles(prior, K)
        new{MMD,typeof(kernel),typeof(prior),typeof(particles)}(kernel,
                                                                epsilon,
                                                                prior, K,
                                                                ones(K)./K,
                                                                particles)
    end
end
"""
    K2ABC(; kernel, epsilon, prior, K)

Creates a `K2ABC` structure.
# Example
```julia
using Distributions
model(x) = [randn() .+ x for _ in 1:3]
data = [[3.], [2.9], [3.3]]
k = K2ABC(kernel = Kernel(),
          epsilon = .1,
          prior = MultivariateNormal([0.], [5.]),
          K = 10^3)
result = run!(k, model, data)
mean(k)
```

[Park, M., Jitkrittum, W., and Sejdinovic, D. (2016),
K2-ABC: Approximate Bayesian Computation with Kernel Embeddings,
Proceedings of Machine Learning Research, 51:398-407
](http://proceedings.mlr.press/v51/park16.html)
"""
K2ABC(; kernel, epsilon, prior, K) = K2ABC{StandardMMD}(kernel = kernel, epsilon = epsilon, prior = prior, K = K)
struct StandardMMD end
struct LinearMMD end
struct RandomFourierMMD end
function mmd(::Type{StandardMMD}, k,
             x::AbstractVector{<:AbstractVector},
             y::AbstractVector{<:AbstractVector})
    nx = length(x)
    ny = length(y)
    1/(nx * (nx - 1)) * sum(i == j ? 0. : k(x[i], x[j]) for i in 1:nx, j in 1:nx) +
    1/(ny * (ny - 1)) * sum(i == j ? 0. : k(y[i], y[j]) for i in 1:ny, j in 1:ny) -
    2/(nx * ny) * sum(k(x[i], y[j]) for i in 1:nx, j in 1:ny)
end
function mmd(::Type{StandardMMD}, k,
             x::AbstractVector{<:Number},
             y::AbstractVector{<:Number})
    -2 * k(x, y)
end


function run!(rng::Random.AbstractRNG, k::K2ABC{MMD}, model, data;
              verbose = true, maxfevals = Inf, callback = () -> nothing) where MMD
    update!(k.kernel, data)
    for i in 1:k.K
        rand!(rng, k.prior, k.particles[i])
        k.weights[i] = exp(-mmd(MMD, k.kernel, model(k.particles[i]), data)/k.epsilon)
    end
    k.weights ./= sum(k.weights)
    (weights = k.weights, particles = k.particles)
end

struct KernelABC{Ke,P,Pa} <: AbstractABC
    kernel::Ke
    prior::P
    K::Int
    delta::Float64
    weights::Vector{Float64}
    particles::Pa
end
"""
    KernelABC(; kernel, prior, K, delta)

Creates a `KernelABC` structure.
# Example
```julia
using Distributions
model(x) = [randn() .+ x for _ in 1:3]
data = [[3.], [2.9], [3.3]]
k = KernelABC(kernel = Kernel(),
              delta = 1e-8,
              prior = MultivariateNormal([0.], [5.]),
              K = 10^3)
result = run!(k, model, data)
mean(k)
```

Fukumizu, K., Song, L. and Gretton, A. (2013),
Kernel Bayes' Rule: Bayesian Inference with Positive Definite Kernels,
Journal of Machine Learning Research, 14:3753-3783,
http://jmlr.org/papers/v14/fukumizu13a.html

See also
[Muandet, K., Fukumizu, K., Sriperumbudur, B. and Schölkopf, B. (2017),
Kernel Mean Embedding of Distributions: A Review and Beyond,
Foundations and Trends® in Machine Learning, 10:1–141
](http://dx.doi.org/10.1561/2200000060)
"""
function KernelABC(; kernel, prior, K, delta)
    KernelABC(kernel, prior, K, delta, ones(K)./K, init_particles(prior, K))
end
function kernelabc!(k, model, data; updatekernel = true)
    y = model.(k.particles)
    updatekernel && update!(k.kernel, y, data)
    kystar = [k.kernel(yi, data) for yi in y]
    G = zeros(length(y), length(y))
    @simd for i in eachindex(y)
        for j in i:length(y)
            @inbounds G[i, j] = G[j, i] = k.kernel(y[i], y[j])
        end
    end
    G = Symmetric(G + k.K*k.delta*I)
    k.weights .= clamp.(G \ kystar, 0, Inf)
end
function run!(rng::Random.AbstractRNG, k::KernelABC, model, data;
              verbose = true, maxfevals = Inf, callback = () -> nothing)
    for i in 1:k.K
        rand!(rng, k.prior, k.particles[i])
    end
    kernelabc!(k, model, data)
    (weights = k.weights, particles = k.particles)
end

