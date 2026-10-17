# How it works

This guide explains what happens under the hood when you create a `Regridder` and use it to regrid data. Understanding these internals can help you make better decisions about grid representations and debug unexpected behavior.

## The high-level picture

When you call `Regridder(dst, src)` on two grids `dst` and `src`, you enter the ConservativeRegridding pipeline.

### 1. Input sanitization and tree construction

First, `dst` and `src` are turned into [GridInterface](interfaces/grids.md) grids via `ConservativeRegridding.asgrid`
(a GridInterface grid is its own; a corner matrix, axis pair or polygon collection becomes one of the shipped grids; the
package extensions answer for their native objects), and each grid's spatial tree is built with
`GridInterface.treeify(grid; leaf_capacity = ForSweeps())`.  The tree:

- Efficiently describes the underlying grid structure,
- Provides a way to efficiently traverse the grid structure in a hierarchical way, from top level (very large) to leaf (each cell).

The tree implements `SpatialTreeInterface`; its leaves emit cell indices, and the grid answers every cell's geometry
(`cell_geometry`) and extent (`cell_extent`) on demand; `dst_areas`/`src_areas` are the `GeometryOps.area` of those geometries.  Structured grids (`GridInterface.Structured`)
get a quadtree cursor over their index ranges, which is what the longitude-latitude, tripolar, cubed sphere, HEALPix, etc.
providers use; any other grid gets an STR R-tree over its cell extents, and a grid can carry its own tree in a `GridAndTree`.

### 2. Finding potentially intersecting cells

Then, we descend simultaneously down into `dst` and `src` trees together, using a dual depth-first tree descent.  
This is an algorithm in the class of _dual-tree_ algorithms, which you can think of as aggressive pruning.  
Both trees are walked simultaneously.  The children of each node are checked for intersection, and pruned if they don't intersect.

As an accelerator, we have an integrated multithreading step here.  Before descending, a short serial pass 
splits the root node pair into a *frontier* of independent node pairs: it repeatedly splits the pair with the 
largest estimated work (overlap area times cell density, from `GridInterface.split_weight`) until there are 
`nthreads * chunks_per_thread` of them.  Each frontier pair then becomes one task, and the tasks' results are 
concatenated in depth-first order, so the threaded search returns exactly what the serial one does.  
No caching need be done ahead of time, since it's all baked into the algorithm.

At the end, we have a list of potentially intersecting cell pairs, represented as a vector of `(src_cell_index, dst_cell_index)` tuples. 
Each index is an integer, starting from 1.  This index corresponds to the linear index of the data corresponding to that cell, in the 
vector (flattened/`vec`) representation of the actual underlying dataset.

### 3. Computing regridding weights

Now that we know which cells potentially intersect, we can compute the regridding weights.  These are stored in a sparse matrix, 
where the row indicates the destination cell index and the column indicates the source cell index.  Each entry is the weight of 
the value of the source cell that contributes to the destination cell.

There are many ways to procure this value.  For now, let's use the simplest method - first order, area based regridding - as an example.
Here, we compute the area of intersection between the source and destination cells, which is then used directly as the weight.  Other methods,
like second-order conservative regridding, bilinear, etc. will use different weight metrics, and some may not even use intersection areas at all.

The way this is done is by splitting up the list of candidates into chunks, and then computing the weights for each chunk in parallel.  The
clipper is chosen from the grids' `cellshape`: when both grids declare `ConvexCells()` we use GeometryOps' optimized spherical convex polygon
intersection (Sutherland-Hodgman); otherwise GeometryOps' general polygon clipper (Foster-Hormann) handles the non-convex cells.  Planar grids
always use the general clipper.  The operator can be overridden by the user (`intersection_operator`).

At the end of this, each task returns a vector of `(dst_cell_index, src_cell_index, weight)` tuples.  These are then concatenated together, and 
assembled into a sparse matrix which is the regridder object.

Success!  We have now constructed a regridder.  There are a couple of steps after this, normalization and so on, but those are more details.

## Customizing the weights: the intersection operator interface

Step 3 above is driven by a single extensible object — the **intersection operator** —
plus four dispatched seams, each with a default that reproduces the area-based weight.
The operator is passed as `Regridder(dst, src; intersection_operator = ...)`, or directly
to [`ConservativeRegridding.intersection_areas`](@ref). The four seams are:

1. **[`IntersectionReturnStyle`](@ref)`(op)`** — how the operator delivers a contribution
   for one work item:
   - [`OutOfPlaceSingleResult`](@ref) (the default): the operator is a function
     `op(src_cell, dst_cell) -> area::Real`, and the driver stores the COO triplet.
   - [`InPlace`](@ref): the operator is
     `op(rows, cols, vals, item, src, dst) -> nothing` (`src` and `dst` the grids) and pushes
     its own COO contributions directly.
2. **[`work_items`](@ref)`(op, candidate_pairs)`** — the units of work the parallel loop
   iterates over (default: one candidate `(src, dst)` pair per unit). Override it to change
   the parallel granularity — e.g. to group all of a source element's candidate cells into
   a single work unit.
3. **[`output_matrix_size`](@ref)`(op, src, dst)`** — the `(nrows, ncols)` shape
   of the assembled matrix (default: `(ncells(dst), ncells(src))`).
4. **[`output_eltype`](@ref)`(op, src, dst)`** — the element type of the sparse
   matrix (default: `Float64`). Override it when you want something other than scalar areas —
   e.g. to build up a matrix of intersection _polygons_ rather than just areas, as the
   shipped [`IntersectionGridOperator`](@ref) does.

Everything else — candidate search, chunking, multithreaded assembly, and sparse-matrix
construction — is shared, so whatever your operator computes runs inside the same parallel
skeleton as the built-in area weights.

### The simple case (`OutOfPlaceSingleResult`)

[`OutOfPlaceSingleResult`](@ref) is the default style, so a custom scalar-weight operator
only needs to be callable; the defaults handle work units and matrix shape:

```julia
using ConservativeRegridding
import GeometryOps as GO

# Reuse the built-in area weight, scaled by a constant factor.
struct ScaledAreaOperator
    factor::Float64
end
(op::ScaledAreaOperator)(src_cell, dst_cell) =
    op.factor * ConservativeRegridding.DefaultIntersectionOperator(GO.Spherical())(src_cell, dst_cell)

regridder = ConservativeRegridding.Regridder(dst, src; intersection_operator = ScaledAreaOperator(2.0))
```

### Emitting a block (`InPlace`)

When one work item should contribute several matrix entries — as the ClimaCore
spectral-element assemblers do, emitting an `Nq²` block per intersection — declare the
operator [`InPlace`](@ref) and `push!` directly onto the COO vectors:

```julia
struct MyBlockOperator
    nrows::Int
    ncols::Int
end
ConservativeRegridding.IntersectionReturnStyle(::MyBlockOperator) = ConservativeRegridding.InPlace()
ConservativeRegridding.output_matrix_size(op::MyBlockOperator, ::Any, ::Any) = (op.nrows, op.ncols)

function (op::MyBlockOperator)(rows, cols, vals, (i_src, i_dst), src, dst)
    src_cell = GridInterface.cell_geometry(src, i_src)
    dst_cell = GridInterface.cell_geometry(dst, i_dst)
    weight = ConservativeRegridding.DefaultIntersectionOperator(GO.Spherical())(src_cell, dst_cell)
    # A real block emitter pushes several (row, col, value) entries per work item;
    # here we push a single one to illustrate the mechanics.
    push!(rows, i_dst)
    push!(cols, i_src)
    push!(vals, weight)
    return nothing
end
```

The ClimaCore extension (`SE → FV` and `FV → SE` regridding) is built entirely on this
interface: each direction is an `InPlace` operator, and `FV → SE` additionally overrides
[`work_items`](@ref) to make the per-element mass-matrix solve the unit of parallel work.

## How we actually do it
