"""
    L(x, y, k) = -log(mean(k(xi, y) for xi in x))
"""
L(x, y, k) = -log(mean(k(xi, y) for xi in x))
"""
    KernelLoss(; loss = L, K = 50,
                 kerneltype = ModifiedGaussian,
                 gamma = .1,
                 heuristic = Nothing,
                 prior = nothing,
                 distance = euclidean,
                 kernel = Kernel(type = kerneltype,
                                 heuristic = heuristic,
                                 gamma = gamma,
                                 distance = distance))

Constructs an structure, used to compare `K` samples generated from a model
to some data with a `kernel` and loss function `loss`. The loss function has the
signature `L(list_of_samples, data, kernel)`; by default it is the negative log
of the average distance between samples and data
(see [LikelihoodfreeInference.L](@ref)).
The `KernelLoss` can be used as a loss in a [`PointEstimator`](@ref).

The returned structure is callable.
# Example
```julia
model(x) = randn() .+ x
data = [2.]
k = KernelLoss()
k([1.], model, data)
```

`K` can be an integer or a function that returns integers, to implement a schedule.
# Example
```julia
using Statistics
model(x) = randn() .+ x
data = [2.]
k = KernelLoss(K = (n) -> n > 20 ? 10^3 : 10)
std(k([1.], model, data, 1) for _ in 1:100)   # large std, because K = 10
std(k([1.], model, data, 21) for _ in 1:100)  # small std, because K = 10^3
```

This `KernelLoss` is inspired by
[Bertl, J., Ewing, G., Kosiol, C., and Futschik, A., (2017)
Approximate maximum likelihood estimation for population genetic inference,
Statistical Applications in Genetics and Molecular Biology. 16:5-6
](http://dx.doi.org/10.1515/sagmb-2017-0016)
"""
struct KernelLoss{P,Tl,TK,Tk}
    loss::Tl
    K::TK
    kernel::Tk
    prior::P
end
function KernelLoss(; loss = L, K = 50,
                      kerneltype = ModifiedGaussian,
                      gamma = .1,
                      heuristic = Nothing,
                      prior = nothing,
                      distance = euclidean,
                      kernel = Kernel(type = kerneltype,
                                      heuristic = heuristic,
                                      gamma = gamma,
                                      distance = distance))
    KernelLoss(loss, K, kernel, prior)
end
computeK(K::Number, ::Any) = K
computeK(K::Function, n) = K(n)
function (k::KernelLoss)(θ, model, data, n = 1)
    S = [model(θ) for _ in 1:computeK(k.K, n)]
    n == 1 && update!(k.kernel, pairwise_euclidean([S..., data]))
    k.loss(S, data, k.kernel) - logpdf(k.prior, θ)
end
mutable struct QDLoss{D,EPS,Tk,P}
    distance::D
    ϵ::EPS
    K::Tk
    prior::P
    mins::Vector{Float64}
end
"""
    QDLoss(; K = 50, epsilon = 2/K, prior = nothing, distance = euclidean)

Constructs an structure, used to compare `K` samples generated from a model
to some data by computing the first epsilon-quantile of the `distance` between
samples and data.
The `QDLoss` can be used as a loss in a [`PointEstimator`](@ref).

The returned structure is callable.
# Example
```julia
model(x) = randn() .+ x
data = [2.]
q = QDLoss()
q([1.], model, data)
```

`K` can be an integer or a function that returns integers, to implement a schedule.
Similarly, `epsilon` can be a number or a function that returns numbers.
# Example
```julia
using Statistics
model(x) = randn() .+ x
data = [2.]
q = QDLoss(K = (n) -> n > 20 ? 10^3 : 10)
std(q([1.], model, data, 1) for _ in 1:100)   # large std, because K = 10
std(q([1.], model, data, 21) for _ in 1:100)  # small std, because K = 10^3
```
"""
function QDLoss(; K = 50, epsilon = 2/computeK(K, 1),
                  prior = nothing, distance = euclidean)
    QDLoss(distance, epsilon, K, prior, Float64[])
end
function lquantile(k::QDLoss{typeof(euclidean)}, S,
                   data::AbstractVector{<:AbstractVector}, n)
    length(k.mins) == 0 && (k.mins = zeros(length(data)))
    ds = [[k.distance(x[i], data[i]) for x in S] for i in eachindex(data)]
    sum(lquantile(ds[i], computeK(k.ϵ, n), Ref(k.mins, i)) for i in eachindex(data))
end
function lquantile(k, S, data, n)
    length(k.mins) == 0 && (k.mins = [0.])
    d = [k.distance(x, data) for x in S]
    lquantile(d, computeK(k.ϵ, n), Ref(k.mins, 1))
end
function lquantile(d, ϵ, min::Base.RefArray)
    q = quantile(d, ϵ)
    q == 0 && return -log(mean(d .== 0)/ϵ) + min[]
    lq = log(q)
    lq < min[] && (min[] = lq)
    lq
end
function (k::QDLoss)(θ, model, data, n = 1)
    S = [model(θ) for _ in 1:computeK(k.K, n)]
    lquantile(k, S, data, n) - 1/length(θ) * logpdf(k.prior, θ)
end
function objective(model, data, loss;
                   callback = (x, y) -> nothing)
    cb = hasmethod(callback, Tuple{Vector{<:Number}, Number}) ?
         callback : (x, y) -> callback()
    n = 0
    θ -> begin
        n += 1
        result = loss(θ, model, data, n)
        cb(θ, result)
        result
    end
end
"""
    PointEstimator(optimizer, loss)
"""
struct PointEstimator{O,L} <: AbstractPointABC
    optimizer::O
    loss::L
end
"""
    PointEstimator(; prior = nothing, lower = nothing, upper = nothing,
                     optimizer = nlopt(prior),
                     losstype = QDLoss, kwargs...)

Creates a `PointEstimator` with given `prior`, `optimizer` and
`loss = losstype(; prior = prior, kwargs...)`. If the prior is `nothing` a
uniform prior is assumed. In this case the optimizer has to be initialized with
bounds, , e.g. `nlopt(lower, upper)` where `lower` and `upper` are arrays.

# Example
```julia
model(x) = randn() .+ x
data = [2.]
p = PointEstimator(lower = [-5], upper = [5], losstype = QDLoss, K = 10^3)
res = run!(p, model, data, maxfevals = 10^5)
```
"""
function PointEstimator(; prior = nothing, lower = nothing, upper = nothing,
                        optimizer = prior === nothing ? (lower === nothing || upper ===nothing) ? error("Please set `prior` or `lower` and `upper`") : nlopt(lower, upper) :
                                                          nlopt(prior),
                          losstype = QDLoss, kwargs...)
    PointEstimator(optimizer, losstype(; prior = prior, kwargs...))
end

function run!(p::PointEstimator, model, data;
              maxfevals = Inf, callback = () -> nothing, verbose = true)
    _optimize(p.optimizer, objective(model, data, p.loss, callback = callback),
              maxfevals = div(maxfevals, computeK(p.loss.K, 1)), verbose = verbose)
end

