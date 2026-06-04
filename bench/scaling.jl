# Conda-free regridder-construction scaling benchmark across grid SIZE, for the two main grid
# families (Oceananigans LatLon, Healpix). Imports NO CairoMakie and NO XESMF, so it runs in a
# minimal environment — this is exactly what the two-ref CI runs, once per ref, in an isolated
# subprocess (see compare.jl), with only the ConservativeRegridding package swapped.
#
#     julia --project=<env> bench/scaling.jl [out.jls] [maxcells]
#
# Emits a Vector of primitive-only NamedTuple rows via `Serialization` (so results round-trip
# across processes / CR versions without deserializing any package-defined types) and prints
# `pathof(ConservativeRegridding)` as a sentinel, so the active CR is provable in the logs.

using ConservativeRegridding
import GeometryOps as GO
import SparseArrays
using Chairmarks
using Serialization

include(joinpath(@__DIR__, "grids.jl"))

# Build once (warm up compilation + record nnz), then time construction with Chairmarks.
# `minimum` is the least-noisy estimator (BenchmarkTools convention) and needs no extra import.
function bench_construction(pair; seconds = 5.0, threaded::Bool = true)
    build() = ConservativeRegridding.Regridder(GO.Spherical(), pair.dst, pair.src;
        normalize = false, threaded = threaded)
    R = build()                                   # warm up + correctness + nnz
    s = @be build() evals = 1 seconds = seconds   # Chairmarks Benchmark
    best = minimum(s)
    return (;
        family     = pair.family,
        method     = threaded ? "CR" : "CR-serial",
        tier       = pair.tier,
        ncells_dst = pair.ncells_dst,
        ncells_src = pair.ncells_src,
        nthreads   = Threads.nthreads(),
        time_s     = best.time,
        allocs     = Int(best.allocs),
        bytes      = Int(best.bytes),
        nnz        = SparseArrays.nnz(R.intersections),
        nsamples   = length(s.samples),
    )
end

function run_scaling(; maxcells = typemax(Int), seconds = 5.0, threaded::Bool = true)
    pairs = vcat(oceananigans_tiers(; maxcells), healpix_tiers(; maxcells))
    rows = NamedTuple[]
    for p in pairs
        @info "scaling" family = p.family tier = p.tier ncells_dst = p.ncells_dst
        push!(rows, bench_construction(p; seconds, threaded))
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    out      = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "results", "scaling.jls")
    maxcells = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : typemax(Int)
    @info "ConservativeRegridding source (sentinel)" path = pathof(ConservativeRegridding) nthreads = Threads.nthreads()
    rows = run_scaling(; maxcells)
    mkpath(dirname(out))
    Serialization.serialize(out, rows)
    @info "wrote scaling results" out nrows = length(rows)
end
