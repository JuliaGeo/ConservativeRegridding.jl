# Microbenchmark for the raster-shaped TopDownQuadtreeCursor hot path.
#
#     julia --project=docs bench/quadtree_cursor.jl
#
# The 896×896 half-cell-shifted grids make cursor child construction visible.

using Chairmarks
import ConservativeRegridding as CR
import Extents
import GeometryOps as GO

# Count the candidate pairs emitted by one complete dual-tree traversal.
mutable struct Counter
    n::Int
end

@inline function (counter::Counter)(::Any, ::Any)
    counter.n += 1
    return nothing
end

function traverse(tree1, tree2)
    counter = Counter(0)
    CR.weighted_dual_depth_first_search(counter, Extents.intersects, tree1, tree2)
    return counter.n
end

function main()
    n = 896
    leaf = 4
    step = 1 / n
    x1 = collect(range(0.0, 1.0; length = n + 1))
    x2 = x1 .+ step / 2
    y = collect(range(0.0, 1.0; length = n + 1))
    grid1 = CR.Trees.RegularGrid(GO.Planar(), x1, y)
    grid2 = CR.Trees.RegularGrid(GO.Planar(), x2, y)
    tree1 = CR.Trees.TopDownQuadtreeCursor(grid1; leafsize = (leaf, leaf))
    tree2 = CR.Trees.TopDownQuadtreeCursor(grid2; leafsize = (leaf, leaf))

    expected = traverse(tree1, tree2)
    sample = @b traverse($tree1, $tree2)
    traverse(tree1, tree2) == expected || error("candidate count changed")

    repo = dirname(dirname(pathof(CR)))
    println((commit = readchomp(`git -C $repo rev-parse HEAD`),
        threads = Threads.nthreads(), shape = (n, n), leafsize = tree1.leafsize,
        candidates = expected, sample))
end

main()
