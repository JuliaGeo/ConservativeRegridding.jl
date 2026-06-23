# Phase-level profiler for Regridder construction on spherical grids — the tool behind
# bench/construction-performance.md. Times each phase separately (treeify → candidate search →
# COO assembly → sparse build → per-cell areas), threaded & serial, at two Oceananigans tiers,
# then prints a statistical Profile of the dominant phase. Run in the bench env:
#
#     julia --project=bench --threads=4 bench/profile_cr.jl
#
# It reaches into the internal phase functions (get_all_candidate_pairs, _parallel_coo, …) so each
# phase can be timed against fixed inputs; the public path is just `Regridder(manifold, dst, src)`.

using ConservativeRegridding
const CR = ConservativeRegridding
const Trees = CR.Trees
import GeometryOps as GO
using SparseArrays
using Profile
using Printf

include(joinpath(@__DIR__, "grids.jl"))

bestof(f, n) = minimum(@elapsed(f()) for _ in 1:n)

const M = GO.Spherical()

function phase_times(Ny; threaded::Bool)
    pair = oceananigans_pair(Ny)
    dst, src = pair.dst, pair.src
    tf = CR.booltype(threaded)
    predicate_f = GO.UnitSpherical._intersects
    op = CR.DefaultIntersectionOperator(M)
    style = CR.IntersectionReturnStyle(op)
    npart = Threads.nthreads() * 4

    dst_tree = Trees.treeify(M, dst)
    src_tree = Trees.treeify(M, src)
    cps = CR.get_all_candidate_pairs(tf, predicate_f, src_tree, dst_tree)
    items = CR.work_items(op, cps)
    nrows, ncols = CR.output_matrix_size(op, src_tree, dst_tree)
    rows, cols, vals = CR._parallel_coo(style, op, items, src_tree, dst_tree, tf; npartitions = npart, progress = false)

    nf = threaded ? 3 : 2   # fewer reps for the slow serial phases
    t_tree = bestof(() -> (Trees.treeify(M, dst); Trees.treeify(M, src)), 3)
    t_cand = bestof(() -> CR.get_all_candidate_pairs(tf, predicate_f, src_tree, dst_tree), nf)
    t_coo  = bestof(() -> CR._parallel_coo(style, op, items, src_tree, dst_tree, tf; npartitions = npart, progress = false), nf)
    t_sp   = bestof(() -> sparse(rows, cols, vals, nrows, ncols), 5)
    t_area = bestof(() -> (CR.areas(M, dst, dst_tree); CR.areas(M, src, src_tree)), 3)
    alloc_coo = @allocated CR._parallel_coo(style, op, items, src_tree, dst_tree, tf; npartitions = npart, progress = false)

    total = t_tree + t_cand + t_coo + t_sp + t_area
    return (; Ny, ncells_dst = nrows, ncells_src = ncols, ncand = length(cps), nnz = length(vals),
        t_tree, t_cand, t_coo, t_sp, t_area, total, alloc_coo)
end

function report(r, label)
    @printf("\n[%s]  Ny=%d  dst=%d src=%d  candidates=%d  nnz=%d\n",
        label, r.Ny, r.ncells_dst, r.ncells_src, r.ncand, r.nnz)
    pct(t) = 100 * t / r.total
    for (name, t) in [("treeify", r.t_tree), ("candidate search", r.t_cand),
                      ("COO assembly", r.t_coo), ("sparse build", r.t_sp), ("per-cell areas", r.t_area)]
        @printf("   %-18s %8.1f ms   %5.1f%%\n", name, t * 1e3, pct(t))
    end
    @printf("   %-18s %8.1f ms\n", "TOTAL", r.total * 1e3)
    @printf("   COO assembly allocs: %.1f MB\n", r.alloc_coo / 2^20)
end

println("Julia threads = ", Threads.nthreads())

for Ny in (64, 128), th in (false, true)   # warm up both code paths at both sizes
    p = oceananigans_pair(Ny)
    CR.Regridder(M, p.dst, p.src; normalize = false, threaded = th)
end

for Ny in (64, 128)
    report(phase_times(Ny; threaded = true),  "threaded Ny=$Ny")
    report(phase_times(Ny; threaded = false), "serial   Ny=$Ny")
end

println("\n\n================ PROFILE: serial Regridder construction, Ny=128 ================")
pair = oceananigans_pair(128)
CR.Regridder(M, pair.dst, pair.src; normalize = false, threaded = false)
Profile.clear()
Profile.init(; n = 10^7, delay = 0.0007)
Profile.@profile for _ in 1:3
    CR.Regridder(M, pair.dst, pair.src; normalize = false, threaded = false)
end
Profile.print(; format = :flat, sortedby = :count, mincount = 25)
