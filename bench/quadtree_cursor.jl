# Microbenchmark for the raster-shaped TopDownQuadtreeCursor hot path.
#
# Run this file with the ConservativeRegridding checkout under test first in
# JULIA_LOAD_PATH. The default 896×896 half-cell-shifted grids make the stock
# cursor's child construction visible while keeping a five-sample run short.

import ConservativeRegridding as CR
import Extents
import GeometryOps as GO

const N = parse(Int, get(ENV, "CR_CURSOR_BENCH_N", "896"))
const LEAF = parse(Int, get(ENV, "CR_CURSOR_BENCH_LEAF", "4"))
const REPS = parse(Int, get(ENV, "CR_CURSOR_BENCH_REPS", "5"))

step = 1 / N
x1 = collect(range(0.0, 1.0; length = N + 1))
x2 = x1 .+ step / 2
y = collect(range(0.0, 1.0; length = N + 1))
grid1 = CR.Trees.RegularGrid(GO.Planar(), x1, y)
grid2 = CR.Trees.RegularGrid(GO.Planar(), x2, y)

prototype1 = CR.Trees.TopDownQuadtreeCursor(grid1)
prototype2 = CR.Trees.TopDownQuadtreeCursor(grid2)
tree1, tree2 = if hasproperty(prototype1, :leafsize)
    (CR.Trees.TopDownQuadtreeCursor(grid1; leafsize = (LEAF, LEAF)),
     CR.Trees.TopDownQuadtreeCursor(grid2; leafsize = (LEAF, LEAF)))
else
    (prototype1, prototype2)
end

mutable struct Counter
    n::Int
end

@inline function (counter::Counter)(::Any, ::Any)
    counter.n += 1
    return nothing
end

function traverse(tree1, tree2)
    counter = Counter(0)
    CR.cached_dual_depth_first_search(counter, Extents.intersects, tree1, tree2)
    return counter.n
end

expected = traverse(tree1, tree2)
samples = NamedTuple[]
for rep in 1:REPS
    GC.gc()
    result = @timed traverse(tree1, tree2)
    result.value == expected || error("candidate count changed")
    push!(samples, (rep = rep, seconds = result.time, bytes = result.bytes,
        gc_seconds = result.gctime, candidates = result.value))
end
best = samples[argmin(getproperty.(samples, :seconds))]

repo = dirname(dirname(pathof(CR)))
println((commit = readchomp(`git -C $repo rev-parse HEAD`),
    threads = Threads.nthreads(), shape = (N, N),
    leafsize = hasproperty(tree1, :leafsize) ? tree1.leafsize : (2, 2),
    reps = REPS, best = best, samples = samples))
