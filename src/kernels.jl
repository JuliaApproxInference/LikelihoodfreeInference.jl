abstract type AbstractBandwidth end
struct Smoothed{B,BB} <: AbstractBandwidth
    base_bandwidth::BB
    γ::B
    α::Float64
end
function update!(s::Smoothed, x)
    b = update!(base_bandwidth, x)
    copyto!(b.γ, s.α * b.γ .+ (1 - b.α) * b)
end
struct Scheduled{B,S} <: AbstractBandwidth
    bandwidth::B
    schedule::S
end
bandwidth(b::Scheduled) = b.schedule(bandwidth(b.bandwidth))
function update!(s::Scheduled, x)
    update!(s.bandwidth, x)
    update!(s.schedule)
    s
end
mutable struct RosenblattSchedule
    n::Int
    exponent::Float64
end
RosenblattSchedule(d) = RosenblattSchedule(0, -1/(d + 4))
(s::RosenblattSchedule)(b) = b * s.n^s.exponent
update!(s::RosenblattSchedule) = s.n += 1
struct Bandwidth{H,B} <: AbstractBandwidth
    heuristic::H
    γ::B
    function Bandwidth(heuristic::H, gamma::T) where {H, T}
        if T <: Number && H <: AbstractHeuristic
            gamma = Ref(gamma)
        end
        new{H, typeof(gamma)}(heuristic, gamma)
    end
end
Bandwidth(; heuristic = MedianHeuristic(), gamma = NaN) = Bandwidth(heuristic, gamma)
Base.copyto!(x::Base.RefValue, y) = x[] = y
bandwidth(b::AbstractBandwidth) = b.γ
bandwidth(b::Bandwidth{<:Any, <:Base.RefValue}) = b.γ[]
abstract type AbstractHeuristic end
struct MedianHeuristic <: AbstractHeuristic
    factor::Float64
end
MedianHeuristic() = MedianHeuristic(1.)
struct ScottsHeuristic <: AbstractHeuristic end
update!(b::Bandwidth{MedianHeuristic}, x) = copyto!(b.γ, b.heuristic.factor * median(x))
update!(b::Bandwidth{ScottsHeuristic}, x) = copyto!(b.γ, std(x))
update!(b::AbstractBandwidth, ::Any) = b

struct Kernel{T,B,F}
    bandwidth::B
    distance::F
end
"""
    Kernel(; gamma = NaN,
             type = Gaussian,
             distance = euclidean,
             heuristic = MedianHeuristic(),
             bandwidth = Bandwidth(heuristic, gamma))

Creates a `Kernel`. See `subtypes(LikelihoodfreeInference.AbstractKernel)`,
`subtypes(LikelihoodfreeInference.AbstractBandwidth)`,
`subtypes(LikelihoodfreeInference.AbstractHeuristic)` for options.
The bandwidth parameter `gamma` can be a number or a vector.

# Example 1
```julia
k = Kernel()
k(2.) == exp(-2^2/2)
using LikelihoodfreeInference: update!, bandwidth  # some internals.
bandwidth(k) === NaN                               # default bandwidth.
update!(k, [2., 2.])                               # median heuristic bandwidth.
bandwidth(k) == 2.
a = [1, 2, 3]; b = [4, 5, 6]
k(a, b) == k(k.distance(a, b, bandwidth(k))) == exp(-sum(abs2, a - b)/(2*bandwidth(k)^2))
```
# Example 2
```julia
k = Kernel(gamma = zeros(2), heuristic = ScottsHeuristic())
data = [2*randn(2) for _ in 1:5]
update!(k, data)
bandwidth(k) == std(data)
a = [1, 2]; b = [4, 5]
k(a, b) ≈ exp(-sum(abs2, (a - b) ./ bandwidth(k))/2)
```
"""
function Kernel(; gamma = NaN,
                  type = Gaussian,
                  distance = euclidean,
                  heuristic = MedianHeuristic(),
                  bandwidth = Bandwidth(heuristic, gamma)
                 )
    Kernel{type, typeof(bandwidth), typeof(distance)}(bandwidth, distance)
end
bandwidth(k::Kernel) = bandwidth(k.bandwidth)
update!(k::Kernel, x) = update!(k.bandwidth, x)
function update!(k::Kernel, simulated::Vector{T}, data::T) where T
    if T <: AbstractVector{<:Number}
        d = [k.distance(x, data) for x in simulated]
    elseif T <: AbstractVector{<:AbstractVector}
        d = vcat([[k.distance(xi, d) for xi in x] for x in simulated, d in data]...)
    end
    update!(k, d)
end
update!(k::Kernel{<:Any,<:Bandwidth{Nothing}}, ::Vector{T}, ::T) where T = k
(k::Kernel)(a, b) = k(k.distance(a, b, bandwidth(k)))

abstract type AbstractKernel end
struct Gaussian <: AbstractKernel end
struct ModifiedGaussian <: AbstractKernel end
(k::Kernel{Gaussian})(x) = exp(-x^2/2)
(k::Kernel{ModifiedGaussian})(x) = exp(-1/2 * (x < 1 ? x^2 : x))
