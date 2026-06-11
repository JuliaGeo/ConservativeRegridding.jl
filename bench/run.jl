# Local entry point: run the full benchmark suite in the bench environment and write graphs +
# a markdown summary to bench/results/.
#
#     julia --project=bench bench/run.jl [maxcells]
#
# Produces, under bench/results/:
#     scaling.png  scaling.jls   — Part 1: construction scaling, Oceananigans + Healpix
#     xesmf.png    xesmf.jls     — Part 2: ConservativeRegridding vs XESMF (Oceananigans)
#
# (The PR-vs-base comparison is compare.jl; it is what the CI runs.)

include(joinpath(@__DIR__, "scaling.jl"))
include(joinpath(@__DIR__, "xesmf.jl"))
include(joinpath(@__DIR__, "plots.jl"))

maxcells = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : typemax(Int)
outdir = joinpath(@__DIR__, "results")
mkpath(outdir)

@info "Part 1: construction scaling" maxcells nthreads = Threads.nthreads()
scaling_rows = run_scaling(; maxcells)
Serialization.serialize(joinpath(outdir, "scaling.jls"), scaling_rows)
plot_scaling(scaling_rows; path = joinpath(outdir, "scaling.png"))

@info "Part 2: ConservativeRegridding vs XESMF"
xesmf_rows = run_xesmf(; maxcells)
Serialization.serialize(joinpath(outdir, "xesmf.jls"), xesmf_rows)
plot_xesmf(xesmf_rows; path = joinpath(outdir, "xesmf.png"))

println("\n## Construction scaling (ConservativeRegridding, threaded)\n")
println(summary_markdown(scaling_rows))
println("\n## ConservativeRegridding vs XESMF (Oceananigans)\n")
println(xesmf_summary_markdown(xesmf_rows))
@info "graphs written" dir = outdir
