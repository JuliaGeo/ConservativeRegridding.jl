using Documenter
using ConservativeRegridding

makedocs(
    sitename = "ConservativeRegridding.jl",
    authors = "Milan Kloewer and contributors",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://juliageo.org/ConservativeRegridding.jl/stable/",
        assets = String[],
    ),
    modules = [ConservativeRegridding],
    pages = [
        "Home" => "index.md",
    ],
    warnonly = [:missing_docs],
)

deploydocs(
    repo = "github.com/JuliaGeo/ConservativeRegridding.jl.git",
    devbranch = "main",
    push_preview = true,
)

