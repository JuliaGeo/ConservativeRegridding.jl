using Documenter
using ConservativeRegridding

makedocs(
    sitename = "ConservativeRegridding.jl",
    authors = "Anshul Singhvi, Milan Kloewer, Simone Silvestri, and contributors",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://juliageo.org/ConservativeRegridding.jl/stable/",
        assets = String[],
    ),
    modules = [ConservativeRegridding],
    pages = [
        "Home" => "index.md",
        "How it works" => "how_it_works.md",
    ],
    warnonly = true,#[:missing_docs],
)

deploydocs(
    repo = "github.com/JuliaGeo/ConservativeRegridding.jl.git",
    devbranch = "main",
    push_preview = true,
)

