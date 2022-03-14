using Documenter, Literate, LikelihoodfreeInference, CMAEvolutionStrategy, BlackBoxOptim, SimultaneousPerturbationStochasticApproximation, BayesianOptimization, StatsPlots

OUTDIR = joinpath(@__DIR__, "src", "generated")
for example in ("toyexample.jl", "blowfly.jl")
    infile = joinpath(@__DIR__, "literate", example)
    Literate.markdown(infile, OUTDIR, documenter = true)
    Literate.notebook(infile, OUTDIR, documenter = true, execute = true)
end

makedocs(
    modules = [LikelihoodfreeInference],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "Johanni Brea",
    sitename = "LikelihoodfreeInference.jl",
    pages = Any[
                "index.md",
                "generated/toyexample.md",
                "generated/blowfly.md",
                "reference.md"
               ]
    # strict = true,
    # clean = true,
    # checkdocs = :exports,
)

deploydocs(
    repo = "github.com/jbrea/LikelihoodfreeInference.jl.git",
    push_preview = true
)
