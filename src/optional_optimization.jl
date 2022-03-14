@require CMAEvolutionStrategy = "8d3b24bd-414e-49e0-94fb-163cc3a3e411" begin
    mutable struct CMA
        x0
        sigma0
        lower
        upper
        kwargs
    end
    cma(prior; kwargs...) = cma(lower(prior), upper(prior); kwargs...)
    """
        cma(lower, upper;
            sigma0 = 0.25,
            kwargs...)
        cma(prior; kwargs...) = pycma(lower(prior), upper(prior); kwargs...)

    CMA optimizer for [`PointEstimator`](@ref), available when
    `using CMAEvolutionStrategy`.
    `kwargs` are passed to `CMAEvolutionStrategy.minimize`.
    """
    function cma(lower, upper;
                 sigma0 = 0.25,
                 kwargs...)
        d = length(lower)
        if d == 1
            error("CMA-ES is not recommended to be used in 1 dimension.")
        end
        CMA(rand(d) .* (upper .- lower) + lower, sigma0,
            lower, upper, kwargs)
    end
    export cma
    function _optimize(o::CMA, f; maxfevals, verbose = 1)
        res = CMAEvolutionStrategy.minimize(f, o.x0, o.sigma0;
                                            lower = o.lower, upper = o.upper,
                                            o.kwargs...)
        (x = xbest(res), f = fbest(res), res = res)
    end
end
@require BlackBoxOptim = "a134a8b2-14d6-55f6-9291-3336d3ab0209" begin
    mutable struct BBO
        options
    end
    bbo(prior; kwargs...) = bbo(lower(prior), upper(prior); kwargs...)
    """
        bbo(lower, upper; kwargs...)
        bbo(prior; kwargs...) = bbo(lower(prior), upper(prior); kwargs...)

    BlackBoxOptim optimizer for [`PointEstimator`](@ref), available
    when `using BlackBoxOptim`. `kwargs` are passed to `BlackBoxOptim.bboptimize`.
    """
    function bbo(lower, upper; kwargs...)
        srange = [(l, u) for (l, u) in zip(lower, upper)]
        BBO((SearchRange = srange, pairs(kwargs)...))
    end
    export bbo
    function _optimize(o::BBO, f; maxfevals, verbose)
        verbose = typeof(verbose) <: Bool ?
                    (verbose ? :verbose : :silent) : verbose
        res = BlackBoxOptim.bboptimize(f; MaxFuncEvals = maxfevals,
                                       TraceMode = verbose,
                                       o.options...)
        (x = BlackBoxOptim.best_candidate(res),
         f = BlackBoxOptim.best_fitness(res),
         res = res)
    end
end
@require BayesianOptimization = "4c6ed407-134f-591c-93fa-e0f7c164a0ec" begin
    const BO = BayesianOptimization
    mutable struct BOTmp
        model
        modeloptimizer
        acquisition
        lb
        ub
        options
        opt
    end
    bo(prior; kwargs...) = bo(lower(prior), upper(prior); kwargs...)
    """
        bo(lower, upper;
                acquisition = BO.UpperConfidenceBound(),
                capacity = 10^3,
                mean = BO.GaussianProcesses.MeanConst(0.),
                kernel = BO.GaussianProcesses.SEArd(zeros(length(lower)), 5.),
                logNoise = 0,
                modeloptimizer_options = NamedTuple(),
                kwargs...)
        bo(prior; kwargs...) = bo(lower(prior), upper(prior); kwargs...)

    BayesianOptimization optimizer for [`PointEstimator`](@ref), available
    when `using BayesianOptimization`. `kwargs` are passed to `BayesianOptimization.BOpt`.
    """
    function bo(lower, upper;
                acquisition = BO.UpperConfidenceBound(),
                capacity = 10^3,
                mean = BO.GaussianProcesses.MeanConst(0.),
                kernel = BO.GaussianProcesses.SEArd(zeros(length(lower)), 5.),
                logNoise = 0,
                modeloptimizer_options = NamedTuple(),
                kwargs...)
        model = BO.GaussianProcesses.ElasticGPE(length(lower),
                           mean = mean,
                           kernel = kernel,
                           logNoise = 0.,
                           capacity = capacity)
        modeloptimizer_options = merge((every = 125,
                                        noisebounds = [-4, 3],
                                        kernbounds = [[log.((upper .- lower)./10^2); 0],
                                                      [log.(2*(upper .- lower)); 10]],
                                        maxeval = 40),
                                       modeloptimizer_options)
        modeloptimizer = BO.MAPGPOptimizer(; modeloptimizer_options...)
        BOTmp(model, modeloptimizer, acquisition, lower, upper,
               merge((repetitions = 1, maxiterations = 500,
                      acquisitionoptions = (restarts = 20, maxtime = 2),
                      sense = BO.Min), kwargs), nothing)
    end
    export bo
    function _optimize(o::BOTmp, f; maxfevals, verbose)
        verbose = verbose ? BO.Progress : BO.Silent
        opt = BO.BOpt(f, o.model, o.acquisition, o.modeloptimizer, o.lb, o.ub;
                   o.options..., verbosity = verbose,
                   maxiterations = maxfevals)
        o.opt = opt
        res = BO.boptimize!(opt)
        (x = res.model_optimizer, f = res.model_optimum, res = opt)
    end
end
@require SimultaneousPerturbationStochasticApproximation = "1663b253-47ef-444c-a4fb-e919f25dc38d" begin
    struct SPSATmp
        s
    end
    spsa(prior; kwargs...) = spsa(lower(prior), upper(prior); kwargs...)
    """
        spsa(lower, upper; kwargs...)
        spsa(prior; kwargs...) = spsa(lower(prior), upper(prior); kwargs...)

    SimultaneousPerturbationStochasticApproximation optimizer for [`PointEstimator`](@ref), available when `using SimultaneousPerturbationStochasticApproximation`. `kwargs` are passed to `SimultaneousPerturbationStochasticApproximation.SPSA`.
    """
    function spsa(lower, upper; kwargs...)
        SPSATmp(SimultaneousPerturbationStochasticApproximation.SPSA(; lower = lower, upper = upper, kwargs...))
    end
    export spsa
    function _optimize(o::SPSATmp, f; maxfevals, verbose)
        res = SimultaneousPerturbationStochasticApproximation.minimize!(o.s, f, maxfevals = maxfevals, verbose = verbose)
        (x = res[1].x, f = res[1].f, res = res)
    end
end
@require CMAEvolutionStrategy = "8d3b24bd-414e-49e0-94fb-163cc3a3e411" begin
    cma(prior; kwargs...) = cma(lower(prior), upper(prior); kwargs...)
    function cma(lower, upper;
                 K = Ref(0), minK = 2, maxK = 10^3, alphaK = 1.5,
                 maxfevals = typemax(Int),
                 kwargs...)
        n = length(lower)
        x0 = (upper .- lower) .* rand(n) .+ lower
        cb = s -> begin
            if s > 0
                if K[] < maxK
                    K[] = min(maxK, round(Int, K[] * alphaK))
                    return true
                else
                    return true
                end
            else
                if K[] > minK
                    K[] = max(minK, round(Int, K[] / alphaK^.25))
                end
                return false
            end
        end
        noise = CMAEvolutionStrategy.NoiseHandling(n, callback = cb)
        CMAEvolutionStrategy.Optimizer(x0, .25; noise_handling = noise,
                                       lower = lower, upper = upper,
                                       maxfevals = maxfevals, kwargs...)
    end
    export cma
    function _optimize(o::CMAEvolutionStrategy.Optimizer, f; maxfevals, verbose)
        o.stop.maxfevals = maxfevals
        o.logger.verbosity = verbose
        res = CMAEvolutionStrategy.run!(o, f)
        (x = CMAEvolutionStrategy.population_mean(res),
         f = CMAEvolutionStrategy.fbest(res), res = res)
    end
end
