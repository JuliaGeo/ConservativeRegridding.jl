# How it works

This guide explains what happens under the hood when you create a `Regridder` and use it to regrid data. Understanding these internals can help you make better decisions about grid representations and debug unexpected behavior.

## The high-level picture

When you call `Regridder(dst, src)` on two grids `dst` and `src`, you enter the ConservativeRegridding pipeline.

### 1. Input sanitization and tree construction

First, `dst` and `src` are converted into spatial tree structures via `Trees.treeify`.  This creates a data structure that:

- Efficiently describes the underlying grid structure,
- Caches expensive computations (like calculating the vertices of each cell, for e.g. tripolar grids),
- Provides a way to efficiently traverse the grid structure in a hierarchical way, from top level (very large) to leaf (each cell).

The object returned by `treeify` must implement the `GeometryOps.SpatialTreeInterface` as well as our internal `Trees` interface.  
Several helper structs, like the quadtree representations and cursors, are made available to help with this, and those are used for e.g. the longitude-latitude, tripolar, cubed sphere, etc. grids.
But an implementor can also create their own tree structures, as is done in the Healpix extension here.

### 2. Finding potentially intersecting cells

Then, we descend simultaneously down into `dst` and `src` trees together, using a dual depth-first tree descent.  
This is an algorithm in the class of _dual-tree_ algorithms, which you can think of as aggressive pruning.  
Both trees are walked simultaneously.  The children of each node are checked for intersection, and pruned if they don't intersect.

As an accelerator, we have an integrated multithreading step here.  Once we have descended low enough on the tree 
for each node that it has enough children, we then spawn out a task for those nodes to continue the search on a different thread.  
This allows us to multithread the search for intersecting cells, without having to manually partition the work.  
No caching need be done ahead of time, since it's all baked into the algorithm.

At the end, we have a list of potentially intersecting cell pairs, represented as a vector of `(dst_cell_index, src_cell_index)` tuples. 
Each index is an integer, starting from 1.  This index corresponds to the linear index of the data corresponding to that cell, in the 
vector (flattened/`vec`) representation of the actual underlying dataset.

### 3. Computing regridding weights

Now that we know which cells potentially intersect, we can compute the regridding weights.  These are stored in a sparse matrix, 
where the row indicates the destination cell index and the column indicates the source cell index.  Each entry is the weight of 
the value of the source cell that contributes to the destination cell.

There are many ways to procure this value.  For now, let's use the simplest method - first order, area based regridding - as an example.
Here, we compute the area of intersection between the source and destination cells, which is then used directly as the weight.  Other methods,
like second-order conservative regridding, bilinear, etc. will use different weight metrics, and some may not even use intersection areas at all.

The way this is done is by splitting up the list of candidates into chunks, and then computing the weights for each chunk in parallel.  We use 
GeometryOps' optimized spherical convex polygon intersection algorithm (Sutherland-Hodgman) by default, but this can be overridden by the user.

At the end of this, each task returns a vector of `(dst_cell_index, src_cell_index, weight)` tuples.  These are then concatenated together, and 
assembled into a sparse matrix which is the regridder object.

Success!  We have now constructed a regridder.  There are a couple of steps after this, normalization and so on, but those are more details.

## Customizing the weights: the intersection operator interface

Step 3 above is driven by a single extensible object — the **intersection operator** —
plus three dispatched seams, each with a default that reproduces the area-based weight.
The operator is passed as `Regridder(dst, src; intersection_operator = ...)`, or directly
to [`ConservativeRegridding.intersection_areas`](@ref). The three seams are:

1. **[`IntersectionReturnStyle`](@ref)`(op)`** — how the operator delivers a contribution
   for one work item:
   - [`OutOfPlaceSingleResult`](@ref) (the default): the operator is a function
     `op(src_cell, dst_cell) -> area::Real`, and the driver stores the COO triplet.
   - [`InPlace`](@ref): the operator is
     `op(rows, cols, vals, item, src_tree, dst_tree) -> nothing` and pushes its own COO
     contributions directly.
2. **[`work_items`](@ref)`(op, candidate_pairs)`** — the units of work the parallel loop
   iterates over (default: one candidate `(src, dst)` pair per unit). Override it to change
   the parallel granularity — e.g. to group all of a source element's candidate cells into
   a single work unit.
3. **[`output_matrix_size`](@ref)`(op, src_tree, dst_tree)`** — the `(nrows, ncols)` shape
   of the assembled matrix (default: destination-cell count × source-cell count).

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

function (op::MyBlockOperator)(rows, cols, vals, (i_src, i_dst), src_tree, dst_tree)
    src_cell = ConservativeRegridding.Trees.getcell(src_tree, i_src)
    dst_cell = ConservativeRegridding.Trees.getcell(dst_tree, i_dst)
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
