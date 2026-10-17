# Grid Interface

ConservativeRegridding regrids [GridInterface](https://github.com/asinghvi17/GridInterface.jl)
grids. `Regridder(dst, src)` turns each input into a grid with
`ConservativeRegridding.asgrid` (a GridInterface grid is its own; a corner
matrix, axis pair or polygon collection becomes one of the shipped grids; the
package extensions answer for their native objects), builds the grids' trees
with `GridInterface.treeify(grid; leaf_capacity = ForSweeps())`, sweeps the
two trees for candidate cell pairs, and clips `cell_geometry(src, i)` against
`cell_geometry(dst, j)` for every candidate.

To use your own grid type, implement GridInterface for it.

## The verbs the regridder uses

```julia
import GridInterface as GrI
import GeometryOpsCore as GOCore

struct MyGrid <: GrI.AbstractGrid ... end

GrI.ncells(g::MyGrid) = <number of cells>
GrI.cell_geometry(g::MyGrid, i::Integer) = <polygon of cell i (1-based)>
GOCore.best_manifold(g::MyGrid) = GOCore.Spherical()   # or Planar()
```

- `ncells` sizes the weight matrix: `(ncells(dst), ncells(src))`, the matrix
  rows and columns being the linear indices of the field data.
- `cell_geometry(grid, i)` is the polygon the clipper sees. Spherical grids use
  `UnitSphericalPoint` vertices, e.g. `GI.Polygon(SA[GI.LinearRing(SA[p1, p2, p3, p4, p1])])`.
- `dst_areas`/`src_areas` are the `GeometryOps.area` of `cell_geometry` on the
  regridder's manifold (not `cell_area`, which a grid may answer analytically),
  so a row's weights sum to exactly its destination cell's area.
- `cell_extent(grid, i)` (a `SphericalCap`, or an `Extents.Extent` on the plane)
  bounds the cell for the tree; its default is the geometry's extent.
- `cellshape(grid)` picks the clipper: single-ring cells (`ConvexCells()` or
  `SimpleCells()`) on both sides select the convex-convex Sutherland-Hodgman
  clip, as the engine has always used; `GeneralCells()` on either side selects
  GeometryOps' general clipper (whose spherical form is not yet exact).
- `treeify(grid; leaf_capacity)` is the tree the sweep walks, a
  `SpatialTreeInterface` tree whose leaves emit cell indices. The default is an
  STR R-tree over the cell extents; structured grids (`GridInterface.Structured`)
  get a quadtree cursor over their index ranges. `split_weight(node)` (the
  number of cells under a node) sizes the threaded frontier.

Cells partition the domain when `tessellation(grid)` is `Tessellating()`;
grids with duplicated cells (the fold row of a U-pivot tripolar grid) say
`Overlapping()` and expose their active cells as a `Subset`.

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
- [ ] `treeify` emits every cell index once (`GridInterface.subtree_indices`)
