# Introduction

Given some measured data `y0` and a potentially stochastic model `model(x)`
that takes parameters `x` and returns simulated data `y`,
`LikelihoodfreeInference.jl` allows to find approximate posterior distributions
over `x` or approximate maximum likelihood (ML) and maximum a posteriori (MAP)
point estimates, by runnning
```julia
run!(method, model, y0)
```
where `method` can be an Approximate Bayesian Computation (ABC) method
`PMC`, `AdaptiveSMC`, `K2ABC`, `KernelABC`
(`subtypes(LikelihoodfreeInference.AbstractABC)`) or
`PointEstimator`, `KernelRecursiveABC`
(`subtypes(LikelihoodfreeInference.AbstractPointABC)`).

## Example

```@example
using LikelihoodfreeInference, StatsPlots, Distributions
gr() # hide
model(x) = randn() .+ x
data = [2.]
method = KernelABC(delta = 1e-8,
                   K = 10^3,
                   kernel = Kernel(),
                   prior = TruncatedMultivariateNormal([0.], [5.],
                                                       lower = [-5.],
                                                       upper = [5.]))
result = run!(method, model, data)
println("Approximate posterior mean = $(mean(method))")
figure = histogram(method, normalize = true, xlims = (-5, 5))
plot!(figure, -1:.01:5, pdf.(Normal.(-1:.01:5, 26/25), 25/26*2.0))
```
