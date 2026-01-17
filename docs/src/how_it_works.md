# How it works

This guide explains what happens under the hood when you create a `Regridder` and use it to regrid data. Understanding these internals can help you make better decisions about grid representations and debug unexpected behavior.

## The high-level picture

When you call `Regridder(dst, src)`, three main things happen:

1. Tree construction: Both grids are converted into spatial tree structures
2. Dual tree query: The trees are walked simultaneously, aggressively pruning branches that can't possibly intersect
3. Weight computation: For the surviving polygon pairs, compute some weight metric for each, and encode that into some operator (usually a sparse matrix).

Step 3 is the one that will change the most across algorithms. 
First-order conservative regridding (the simplest) would use the intersection areas directly as weights.
But other algorithms like second-order conservative regridding, bilinear, etc. will use different weight metrics.

Let's explore each step.

## Step 1: Converting grids to trees

The first thing ConservativeRegridding does is call `Trees.treeify(manifold, grid)` on both the source and destination grids. This function takes whatever grid representation you provide and wraps it in a spatial tree structure.

### Why trees?

A naive approach to finding intersecting polygons would check every source cell against every destination cell — that's `O(n × m)` comparisons. For a 1000×1000 grid against another 1000×1000 grid, that's a trillion polygon intersection checks.

Trees let us do much better. By organizing polygons hierarchically (grouping nearby cells together), we can quickly reject entire regions of one grid that can't possibly intersect with a region of the other grid.  

### How treeify works

The `treeify` function examines your input and chooses an appropriate representation:

| Input type | Tree structure | Notes |
|------------|----------------|-------|
| Matrix of polygons | `ExplicitPolygonGrid` → `TopDownQuadtreeCursor` | Most flexible, least optimized |
| Matrix of corner points | `CellBasedGrid` → `TopDownQuadtreeCursor` | Builds polygons on-the-fly from corners |
| Tuple of 1D coordinate vectors | `RegularGrid` → `TopDownQuadtreeCursor` | For regular lon/lat grids |
| Iterable of polygons | `FlatNoTree` | No spatial indexing, used for small grids |
| Existing spatial tree | Pass-through | If it already implements the tree interface |

A key abstraction is `AbstractCurvilinearGrid`, which represents grids where spatial neighbors in the grid are also neighbors in the underlying matrix. 
This structure enables efficient quadtree decomposition and encodes the structure of the grid in an efficient way.  
Other grids will have different structures, for example Healpix and other hierarchical [global grids](https://en.wikipedia.org/wiki/Discrete_global_grid) - 
which you can use here also.

### Grid representations encode knowledge

Different grid representations encode different facts that can be exploited for performance:

- **`RegularGrid`**: Knows the grid is axis-aligned and regular, so computing bounding boxes for any region is `O(1)` — just index into the coordinate vectors.

- **`CellBasedGrid`**: Stores corner vertices rather than full polygons. Bounding boxes are computed from the corner points of a region, avoiding polygon traversal.

- **`ExplicitPolygonGrid`**: The most general case. Must iterate through polygons to compute bounding boxes, but works with any polygon shape.

The `TopDownQuadtreeCursor` wraps any of these grid types and provides the tree traversal interface. It tracks which rectangular region of the grid it currently represents and can split that region into four quadrants.

## Step 2: The dual tree query

Once both grids have tree representations, ConservativeRegridding performs a *dual depth-first search*. This is the key algorithm that makes conservative regridding tractable for large grids.

### The core idea

The algorithm walks down both trees simultaneously. At each step, it has a node from the source tree and a node from the destination tree. Each node represents some region of its respective grid and has an *extent* (bounding box or spherical cap) that encloses all polygons in that region.

The question at each step is: **can any polygon in this source region possibly intersect any polygon in this destination region?**

If the extents don't intersect, the answer is definitively *no* — and we can prune both branches entirely. This single check eliminates potentially millions of polygon pairs from consideration.

If the extents *do* intersect, we recurse: we check each child of the source node against each child of the destination node. This continues until we reach leaf nodes, at which point we record the polygon index pairs for later intersection computation.

### Pruning in action

Consider two 100×100 grids that only overlap in a corner:

```
Source grid
┌──────────┐
│          │
│          │
│       ┌──┼──────────┐
└───────┼──┘          │
        │↖ overlap    │
        │             │
        │  Dest grid  │
        └─────────────┘
```

At the top level, both trees represent their entire grids. The extents intersect (there is overlap), so we recurse.

Each tree splits into four quadrants. The source's top-left, top-right, and bottom-left quadrants don't intersect with any destination quadrant. Only the source's bottom-right quadrant intersects with the destination's top-left quadrant. We've just eliminated 75% of the source grid and 75% of the destination grid from consideration.

This pruning compounds at each level. If the overlap region is small, the algorithm quickly narrows down to just the relevant cells.

### Multithreading

For large grids, the dual tree query can spawn multiple threads. Once nodes are small enough (determined by an area criterion), each thread independently performs the depth-first search on its subtree pair.

The result is a list of `(source_index, destination_index)` pairs — the candidate polygon pairs that might actually intersect.

## Step 3: Computing intersection weights (first-order conservative)

With the candidate pairs identified, we now compute the actual intersection areas. For each pair:

1. Retrieve the source and destination polygons using `getcell`
2. Compute their geometric intersection
3. Calculate the area of that intersection

### Manifold-aware algorithms

The intersection algorithm depends on the manifold:

- **Planar**: Uses the Foster-Hormann clipping algorithm, which handles arbitrary (including non-convex) polygons
- **Spherical**: Uses Sutherland-Hodgman clipping specialized for convex polygons on the unit sphere

Areas are computed in the appropriate geometry — Cartesian area for planar, spherical excess for spherical manifolds.

### Parallel computation

The list of candidate pairs is partitioned across threads. Each thread independently computes intersection areas for its partition, then results are merged into a sparse matrix.

### The sparse matrix

The final output is a sparse matrix `A` where `A[i,j]` is the intersection area between destination cell `i` and source cell `j`. Most entries are zero (cells don't intersect), which is why sparse storage is essential.

This matrix, combined with cell areas, is everything needed for conservative regridding:

```
destination_field = (A * source_field) ./ destination_areas
```

## Putting it together

Here's the full flow when you call `Regridder(dst, src)`:

```
dst, src
    │
    ▼ treeify()
dst_tree, src_tree
    │
    ▼ dual_depth_first_search()
candidate_pairs: [(i₁,j₁), (i₂,j₂), ...]
    │
    ▼ compute_intersection_areas() [parallel]
(row_indices, col_indices, areas)
    │
    ▼ sparse()
intersection_matrix A
    │
    ▼ + cell areas
Regridder
```

## Tips for users


**Reuse the Regridder**: Building a `Regridder` is expensive; using it is cheap. If you're regridding multiple fields between the same grids, build the regridder once and reuse it.

**Let treeify do its job**: In most cases, you can pass your grid directly to `Regridder` and `Trees.treeify` will choose a reasonable representation. You as the owner of a grid type only need to implement `Trees.treeify` correctly.

**Understand the cost model**: The expensive part is usually the polygon intersections, not the tree query. If regridding is slow, it may be because many polygon pairs actually intersect, not because the tree traversal is inefficient.  To be sure of this, you can time how long it takes to run e.g.
```julia
import GeometryOps as GO

tree1 = ...
tree2 = ...

idxs = Tuple{Int, Int}[]
@time GO.SpatialTreeInterface.dual_depth_first_search(
    GO.UnitSpherical._intersects, tree1, tree2
    ) do i1, i2
    push!(idxs, (i1, i2))
end
```
and see how long that takes (this is singlethreaded).  You can try the same for `ConservativeRegridding.multi

