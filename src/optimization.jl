mutable struct NLOpt
    opt
    x0
end
"""
    nlopt(lower, upper;
          x0 = rand(length(lower)) .* (upper .- lower) .+ lower,
          method = :GN_ESCH, kwargs...)

Resonable methods are `:GN_CRS2_LM`, `:GN_ESCH`, `:GN_ISRES`, `:LN_SBPLX`.
Other options like `maxtime` etc. can be set as keyword arguments
(see https://github.com/JuliaOpt/NLopt.jl).
"""
nlopt(prior; kwargs...) = nlopt(lower(prior), upper(prior); kwargs...)
function nlopt(lower, upper;
               x0 = rand(length(lower)) .* (upper .- lower) .+ lower,
               method = :GN_ESCH, kwargs...)
    D = length(lower)
    opt = NLopt.Opt(method, D)
    for (option, value) in pairs(kwargs)
        setproperty!(opt, option, value)
    end
    NLopt.lower_bounds!(opt, lower)
    NLopt.upper_bounds!(opt, upper)
    NLOpt(opt, x0)
end
export nlopt
function _optimize(o::NLOpt, f; maxfevals, verbose)
    o.opt.min_objective = (x, g) -> f(x)
    o.opt.maxeval = maxfevals
    res = NLopt.optimize(o.opt, o.x0)
    (x = res[2], f = res[1], res = res)
end
