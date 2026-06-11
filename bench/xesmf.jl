# ConservativeRegridding-vs-XESMF regridder CONSTRUCTION-time comparison, Oceananigans-only
# (XESMF only supports Oceananigans fields via its extension). For each tier it times three
# constructors:
#
#     CR          ConservativeRegridding.Regridder(...; threaded = true)   — pure-Julia assembly
#     CR-serial   ConservativeRegridding.Regridder(...; threaded = false)  — single-threaded CR
#     XESMF       XESMF.Regridder(dst_field, src_field; method)            — ESMF weight gen (Python)
#
# CR runs multithreaded by default; XESMF's weight generation is ESMF-controlled (typically
# serial here). We therefore report BOTH CR threaded and CR serial so the comparison isn't a
# misleading N-thread-vs-1-thread chart, and stamp `nthreads` into every row. XESMF's first call
# pays PythonCall import + esmpy/ESMF init, so each workload is warmed up (built once, discarded)
# before timing.
#
#     julia --project=bench bench/xesmf.jl [out.jls] [maxcells]

using ConservativeRegridding
import GeometryOps as GO
import SparseArrays
using Serialization
using Oceananigans: CenterField
using XESMF

include(joinpath(@__DIR__, "grids.jl"))
include(joinpath(@__DIR__, "timing.jl"))

function bench_cr(pair; threaded::Bool, reps::Int = 5, seconds_each::Float64 = 0.5)
    build() = ConservativeRegridding.Regridder(GO.Spherical(), pair.dst, pair.src;
        normalize = false, threaded = threaded)
    R = build()
    t = timed_summary(build; reps, seconds_each)
    return (; family = "Oceananigans", method = threaded ? "CR" : "CR-serial", tier = pair.tier,
        ncells_dst = pair.ncells_dst, ncells_src = pair.ncells_src, nthreads = Threads.nthreads(),
        t.time_s, t.time_lo, t.time_hi, t.nsamples, t.allocs, t.bytes,
        nnz = SparseArrays.nnz(R.intersections))
end

function bench_xesmf(pair; reps::Int = 5, seconds_each::Float64 = 0.5)
    dstf = CenterField(pair.dst)
    srcf = CenterField(pair.src)
    build() = XESMF.Regridder(dstf, srcf; method = "conservative")
    R = build()                                   # warm up: Python import + ESMF init + this shape
    t = timed_summary(build; reps, seconds_each)
    return (; family = "Oceananigans", method = "XESMF", tier = pair.tier,
        ncells_dst = pair.ncells_dst, ncells_src = pair.ncells_src, nthreads = Threads.nthreads(),
        t.time_s, t.time_lo, t.time_hi, t.nsamples, t.allocs, t.bytes,
        nnz = SparseArrays.nnz(R.weights))
end

function run_xesmf(; maxcells = typemax(Int), reps::Int = 5, seconds_each::Float64 = 0.5)
    rows = NamedTuple[]
    for pair in oceananigans_tiers(; maxcells)
        @info "xesmf-compare" tier = pair.tier ncells_dst = pair.ncells_dst
        push!(rows, bench_cr(pair; threaded = true, reps, seconds_each))
        push!(rows, bench_cr(pair; threaded = false, reps, seconds_each))
        push!(rows, bench_xesmf(pair; reps, seconds_each))
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    out      = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "results", "xesmf.jls")
    maxcells = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : typemax(Int)
    @info "ConservativeRegridding source (sentinel)" path = pathof(ConservativeRegridding) nthreads = Threads.nthreads()
    rows = run_xesmf(; maxcells)
    mkpath(dirname(out))
    Serialization.serialize(out, rows)
    @info "wrote xesmf results" out nrows = length(rows)
end
