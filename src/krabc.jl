struct KernelRecursiveABC{Kx,K,HO} <: AbstractPointABC
    kernelABC::K
    kernelx::Kx
    herding_options::HO
    theta::Vector{Float64}
end
"""
    KernelRecursiveABC(; kernel, prior, K, delta = 1e-2,
                         herding_options = (maxtime = 10,
                                            method = :LD_LBFGS,
                                            restarts = 10),
                         kernelx = Kernel())

Creates a `KernelRecursiveABC` structure. `KernelRecursiveABC` iterates a
[`KernelABC`](@ref) step and a kernel herding step. `kernel`, `prior`, `K` and
`delta` determine the `KernelABC` step. `herding_options` options and `kernelx`
determine the herding step.

# Example
```julia
using Distributions
model(x) = [randn() .+ x for _ in 1:3]
data = [[3.], [2.9], [3.3]]
k = KernelRecursiveABC(kernel = Kernel(),
                       delta = 1e-8,
                       prior = MultivariateNormal([0.], [5.]),
                       K = 10^2)
result = run!(k, model, data, maxfevals = 10^3)
result.x
```

[Kajihara, T., Kanagawa, M., Yamazaki, K. and Fukumizu, K. (2018),
Kernel Recursive ABC: Point Estimation with Intractable Likelihood,
Proceedings of the 35th International Conference on Machine Learning, 80:2400-2409
](http://proceedings.mlr.press/v80/kajihara18a.html)
"""
function KernelRecursiveABC(; kernel, prior, K,
                            herding_options = NamedTuple(),
                            delta = 1e-2, kernelx = Kernel())
    KernelRecursiveABC(KernelABC(kernel = kernel, prior = prior, K = K, delta = delta),
                       kernelx,
                       merge((maxtime = 10, method = :LD_LBFGS, restarts = 10),
                             herding_options),
                       zeros(length(prior)))
end

function wrap_gradient(f)
    (x, g) -> begin
        if length(g) > 0
            res = DiffResults.DiffResult(0., g)
            ForwardDiff.gradient!(res, f, x)
            res.value
        else
            f(x)
        end
    end
end
function herding_objective(w, k, θ, newθ, idx)
    wrap_gradient(x -> begin
        res = sum(w[i] * k(x, θ[i]) for i in eachindex(θ))
        if idx > 0
            res -= 1/(idx + 1) * sum(k(x, newθ[i]) for i in 1:idx)
        end
        res
    end)
end
function herding(θ, w, k;
                 d = length(θ[1]),
                 lower = fill(-Inf, d),
                 upper = fill(Inf, d),
                 onlyfirst = false,
                 maxtime = 3,
                 xtol_rel = 1e-4,
                 method = :LD_LBFGS,
                 verbose = false,
                 restarts = 5)
    newθ = deepcopy(θ)
    opt = Opt(method, d)
    opt.lower_bounds = lower
    opt.upper_bounds = upper
    opt.maxtime = maxtime
    opt.xtol_rel = xtol_rel
    x0 = sum(w .* θ)
    for i in eachindex(θ)
        opt.max_objective = herding_objective(w, k, θ, newθ, i-1)
        maxmaxf = -Inf
        for j in 1:restarts
            maxf, maxx, ret = optimize(opt, clamp.(j == 1 ? x0 : θ[wsample(w)] .+ bandwidth(k)/10 * randn(d), lower, upper))
            verbose && ret != :SUCCESS && @show i maxf maxmaxf maxx θ[i] ret opt.numevals
            if maxf > maxmaxf
                maxmaxf = maxf
                newθ[i] .= maxx
            end
        end
        onlyfirst && return newθ[1]
    end
    newθ
end

function run!(rng::Random.AbstractRNG, k::KernelRecursiveABC, model, data;
              verbose = true, maxfevals,
              callback = () -> nothing)
    start_time = now()
    N = div(maxfevals, k.kernelABC.K)
    verbose && @printf "%s %15.s %10.s\n" "iteration" "elapsed" "fevals"
    particles = k.kernelABC.particles
    for i in 1:k.kernelABC.K
        rand!(rng, k.kernelABC.prior, particles[i])
    end
    update!(k.kernelx, pairwise_euclidean(particles))
    kernelabc!(k.kernelABC, model, data)
    k.theta .= mean(k.kernelABC)
    callback()
    verbose && @printf "%9.s %15.s %10.s\n" 1 round(now() - start_time, Second) k.kernelABC.K
    for i in 2:N
        newparticles = herding(particles, k.kernelABC.weights, k.kernelx;
                               lower = lower(k.kernelABC.prior),
                               upper = upper(k.kernelABC.prior),
                               k.herding_options...)
        for j in eachindex(particles)
            particles[j] .= newparticles[j]
        end
        k.theta .= newparticles[1]
        kernelabc!(k.kernelABC, model, data, updatekernel = false)
        callback()
        verbose && @printf "%9.s %15.s %10.s\n" i round(now() - start_time, Second) i * k.kernelABC.K
    end
    (x = herding(particles, k.kernelABC.weights, k.kernelx;
                 onlyfirst = true,
                 lower = lower(k.kernelABC.prior),
                 upper = upper(k.kernelABC.prior),
                 k.herding_options...),)
end

