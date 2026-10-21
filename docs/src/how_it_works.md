# How it works

```@meta
CurrentModule = ConservativeRegridding
```

This guide explains what happens when you create a `Regridder` and use it to regrid data. Knowing the pipeline helps you choose grid representations and debug unexpected behaviour.

## The high-level picture

`Regridder(dst, src)` runs three steps: it turns the two inputs into grids with trees, sweeps the two trees for the cell pairs that may overlap, and assembles the weight of every pair into a sparse matrix.

### 1. Input sanitization and tree construction

`dst` and `src` become [GridInterface](interfaces/grids.md) grids through [`asgrid`](@ref): a GridInterface grid is its own; a corner matrix, axis pair or polygon collection becomes one of the shipped grids; the package extensions answer for their native objects. Each grid's tree is then built with `GridInterface.treeify(grid; leaf_capacity = ForSweeps())`.

The tree is a `SpatialTreeInterface` tree whose leaves emit cell indices. It answers the traversal questions (children, node extents, leaf entries) and the grid answers everything about a cell (`cell_geometry`, `cell_extent`), so `dst_areas`/`src_areas` are the `GeometryOps.area` of exactly the polygons the clipper sees. Which tree a grid gets:

| grid | tree |
|---|---|
| structured (`GridInterface.Structured`: the longitude-latitude, tripolar, cubed-sphere, HEALPix, ... providers) | a quadtree cursor over its index ranges |
| a `Subset` of a grid (the wet cells of an ocean grid, the unique cells of a folded layout) | the parent's tree restricted by `GridInterface.subtree`: the same nodes and extents, children whose `position_span` misses every member dropped, leaves emitting members only |
| `GridAndTree(grid, tree)` | the stored `tree` |
| anything else | an STR R-tree over the cell extents |

A cell index is the linear index of that cell's value in the field (the `vec` of the underlying array), so the weight matrix is indexed by field positions directly.

### 2. Finding potentially intersecting cells

Both trees are walked together by a dual depth-first search ([`weighted_dual_depth_first_search`](@ref ConservativeRegridding.weighted_dual_depth_first_search)): a node pair whose extents do not intersect is pruned with its whole subtree, and every surviving leaf pair yields its cell pairs. One side opens per step:

- the non-leaf side, when only one side can still descend;
- otherwise the side with the greater `SpatialTreeInterface.split_weight` (the cells under the node), the source on a tie.

The extent of each opened child is read once and carried through the descent below it. Once a side reaches a leaf, its entries (`SpatialTreeInterface.child_indices_extents`, the leaf's cell indices and extents) are built once and carried while the other side finishes descending. At a leaf pair the loop runs the source entries outside and the destination entries inside, indexing the destination's entry vector, and emits `(src_cell_index, dst_cell_index)` for every pair whose cell extents intersect.

The search is multithreaded by a *frontier*: a short serial pass splits the root pair, always the pair with the largest estimated work (overlap area times cell density, from `split_weight`), until there are at least `nthreads * chunks_per_thread` independent node pairs. Each frontier pair becomes one task, and the tasks' results are concatenated in depth-first order. The frontier opens the same side the serial search would, so the threaded search returns exactly what the serial one does, in the same order.

At the end, we have the candidate pairs as a vector of `(src_cell_index, dst_cell_index)` tuples. Every pair that truly overlaps is listed; a listed pair need not overlap.

### 3. Computing regridding weights

Now that we know which cells potentially intersect, we can compute the regridding weights.  These are stored in a sparse matrix,
where the row indicates the destination cell index and the column indicates the source cell index.  Each entry is the weight of
the value of the source cell that contributes to the destination cell.

There are many ways to procure this value.  For now, let's use the simplest method - first order, area based regridding - as an example.
Here, we compute the area of intersection between the source and destination cells, which is then used directly as the weight.  Other methods,
like second-order conservative regridding, bilinear, etc. will use different weight metrics, and some may not even use intersection areas at all.

The list of candidates is split into chunks, and the weights of each chunk are computed in parallel. The clipper follows the
manifold and the grids' `cellshape` ([`DefaultIntersectionOperator`](@ref)): on the sphere, convex-convex Sutherland-Hodgman when
both grids have single-ring cells (`ConvexCells()` or `SimpleCells()`), and GeometryOps' general Foster-Hormann clipper when either
has `GeneralCells()`; on the plane, Foster-Hormann always. Pass `intersection_operator` to override it.

At the end of this, each task returns a vector of `(dst_cell_index, src_cell_index, weight)` tuples.  These are then concatenated together, and 
assembled into a sparse matrix which is the regridder object.

Success!  We have now constructed a regridder.  There are a couple of steps after this, normalization and so on, but those are more details.

## Customizing the weights: the intersection operator interface

Step 3 above is driven by a single extensible object — the **intersection operator** —
plus five dispatched seams, each with a default that reproduces the area-based weight.
The operator is passed as `Regridder(dst, src; intersection_operator = ...)`, or directly
to [`ConservativeRegridding.intersection_areas`](@ref). The five seams are:

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
5. **[`task_local_operator`](@ref)`(op)`** — the operator instance one assembly task uses
   (default: `op` itself). Override it to give each task a private mutable cache.

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
import GridInterface

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
