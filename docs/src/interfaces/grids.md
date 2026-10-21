# Grid Interface

ConservativeRegridding regrids [GridInterface](https://github.com/JuliaGeo/GridInterface.jl)
grids. `Regridder(dst, src)` turns each input into a grid with
[`asgrid`](@ref) (a GridInterface grid is its own; a corner matrix, axis pair
or polygon collection becomes one of the shipped grids; the package extensions
answer for their native objects), builds each grid's tree with
`GridInterface.treeify(grid; leaf_capacity = ForSweeps())`, sweeps the two
trees for candidate cell pairs, and clips `cell_geometry(src, i)` against
`cell_geometry(dst, j)` for every candidate.

To regrid your own grid type, implement GridInterface for it.

## The verbs the regridder uses

```julia
import GridInterface as GrI
import GeometryOpsCore as GOCore

struct MyGrid <: GrI.AbstractGrid ... end

GrI.ncells(g::MyGrid) = <number of cells>
GrI.cell_geometry(g::MyGrid, i::Integer) = <polygon of cell i (1-based)>
GOCore.best_manifold(g::MyGrid) = GOCore.Spherical()   # or Planar()
```

| verb | what the regridder does with it |
|---|---|
| `ncells(grid)` | sizes the weight matrix, `(ncells(dst), ncells(src))`; rows and columns are the linear indices of the field data |
| `cell_geometry(grid, i)` | the polygon the clipper sees; spherical grids use `UnitSphericalPoint` vertices, e.g. `GI.Polygon(SA[GI.LinearRing(SA[p1, p2, p3, p4, p1])])` |
| `best_manifold(grid)` | the manifold areas and intersections are measured on; both grids must agree, or the manifold is passed as `Regridder(manifold, dst, src)` |
| `cell_extent(grid, i)` | bounds the cell for the tree: a `SphericalCap`, or an `Extents.Extent` on the plane; default: the geometry's extent |
| `cellshape(grid)` | picks the clipper: single-ring cells (`ConvexCells()` or `SimpleCells()`) on both sides select the convex-convex Sutherland-Hodgman clip; `GeneralCells()` on either side selects GeometryOps' general clipper |
| `treeify(grid; leaf_capacity)` | the tree the sweep walks ([below](@ref "The tree")) |

`dst_areas`/`src_areas` are the `GeometryOps.area` of `cell_geometry` on the
regridder's manifold, so a row's weights sum to exactly its destination cell's
area. A grid's own `cell_area` may be analytic (a HEALPix pixel's) and differ
from its polygon's area; the regridder reads the polygon.

Cells partition the domain when `tessellation(grid)` is `Tessellating()`;
grids with duplicated cells (the fold row of a U-pivot tripolar grid) say
`Overlapping()` and expose their active cells as a `Subset`.

## The tree

`treeify(grid; leaf_capacity = ForSweeps())` returns a `SpatialTreeInterface`
tree whose leaves emit every cell index exactly once. The sweep asks it the
four traversal questions (`isleaf`, `getchild`, `node_extent`,
`child_indices_extents`) and `split_weight(node)`, the cells under a node,
which decides which side of a node pair opens first and sizes the threaded
frontier. Which tree a grid gets:

- a structured grid (`GridInterface.Structured`) gets a quadtree cursor over
  its index ranges, with `leaf_capacity(grid, ForSweeps())` cells per leaf (16
  by default);
- a `Subset` gets its parent's tree restricted to its members
  ([below](@ref "Restricting a grid"));
- a `GridAndTree` answers its stored tree;
- any other grid gets an STR R-tree over its cell extents, built on each call.

A tree whose `node_extent` computes an extent rather than reading a stored one
declares `SpatialTreeInterface.node_extent_is_expensive(::Type{MyNode}) = true`
on its node type; GridInterface's cursors do, and its restricted cursors
forward it. CR's sweep reads each opened child's extent once per visit and
carries it down the descent; the trait serves consumers that cache child
extents, such as `SpatialTreeInterface.dual_depth_first_search`.

## Restricting a grid

Regrid part of a grid (the wet cells of an ocean grid, one region's cells) by
passing a `Subset`: `GridInterface.subset(grid, cells)` for strictly ascending
positions `cells`. A subset is a grid in its own right with positions
`1:length(cells)`, so the weight matrix is sized to the members, and its tree
is the parent's own tree restricted by `GridInterface.subtree(grid, cells)`:
the same nodes and extents, children whose `position_span` (a bound on the
parent positions under a node) misses every member dropped, leaves emitting
members only. A custom tree takes part by answering `position_span(node)`; a
tree that cannot say cheaply answers `nothing` and is filtered at its leaves,
and a grid with no tree of its own gets an R-tree over the members.

## Structured grids

For a lattice of quadrilaterals (one or several faces), subtype
`GridInterface.Structured.AbstractStructuredGrid` and implement its face hooks
(`nfaces`, `facesize`, `facevertex`, `faceposition`, `facecell`, ...): the
quadtree cursor, extents, neighbours and point location follow. The shipped
`CellBasedGrid` (a corner matrix), `RegularGrid` (two axis vectors) and
`ExplicitPolygonGrid` (a polygon matrix) are examples, as are the providers in
the Oceananigans, ClimaCore, Healpix and RingGrids extensions.

## Supplying a tree

Wrap a grid with a tree you already have in `GridInterface.GridAndTree(grid, tree)`:
`treeify` then answers that tree. Any `SpatialTreeInterface` tree whose leaves
emit the grid's cell indices works (an `STRtree` of the polygons, say).

## Checklist

- [ ] `ncells` matches the field's linear index domain
- [ ] `cell_geometry` winds counter-clockwise with the right vertex type
- [ ] `best_manifold` returns the right manifold
- [ ] `cellshape` says `ConvexCells()` only for convex cells
- [ ] `tessellation` says `Overlapping()` when cells duplicate
- [ ] `treeify` emits every cell index once (`GridInterface.subtree_indices`)
