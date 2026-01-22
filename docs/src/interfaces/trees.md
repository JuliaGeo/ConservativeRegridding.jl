# Tree Interface

To use a grid with ConservativeRegridding.jl, implement two interfaces:

1. **GeometryOps SpatialTreeInterface** - enables efficient spatial search via tree traversal
2. **ConservativeRegridding.Trees interface** - provides access to cell geometry

## SpatialTreeInterface

The regridder finds intersecting cell pairs using dual depth-first search on two spatial trees. Each node needs:

- **Bounding extent** - a SphericalCap (or Extent for planar) that contains all descendant cells. Used to prune branches that can't intersect.
- **Children** - either child nodes to descend into, or leaf status indicating this is a grid cell.
- **Leaf identity** - for leaf nodes, the index of the corresponding grid cell.

```julia
import GeometryOps: SpatialTreeInterface as STI

STI.isspatialtree(::Type{<:MyTree}) = true
STI.isleaf(node) = <true if this node represents a grid cell>
STI.nchild(node) = <number of children, 0 for leaves>
STI.getchild(node, i::Int) = <i-th child node (1-based)>
STI.node_extent(node) = <SphericalCap bounding all descendants>

# For leaf nodes: return (cell_index, extent) pairs
STI.child_indices_extents(node) = ((cell_index, extent),)
```

For quadrilateral cells, use `Trees.circle_from_four_corners(corners, ())` to compute the SphericalCap from 4 UnitSphericalPoints.

## Trees Interface

The regridder needs to access cell geometry for intersection computation:

```julia
using ConservativeRegridding.Trees

Trees.ncells(tree) = <total number of grid cells>
Trees.getcell(tree, i::Int) = <polygon for cell i (1-based)>
Trees.getcell(tree) = <iterator over all cell polygons>
Trees.treeify(::GO.Spherical, tree) = tree  # passthrough if already a tree

GOCore.best_manifold(tree) = GO.Spherical()  # or GO.Planar()
```

Cell polygons should use `UnitSphericalPoint` vertices for spherical grids:
```julia
GI.Polygon(SA[GI.LinearRing(SA[p1, p2, p3, p4, p1])])
```

## Hierarchical Grids

For grids with natural hierarchy (HEALPix, cubed sphere, etc.), the tree structure mirrors the grid's refinement hierarchy. You may need separate types for the root node (which may have different branching) and interior nodes.

The key insight: `getchild` encodes the parent-child relationship of your grid's hierarchy. The tree doesn't store the full structure—it computes children on demand from the node's position in the hierarchy.

## Checklist

- [ ] `node_extent` returns tight bounding SphericalCap
- [ ] `isleaf` correctly identifies leaf level
- [ ] `child_indices_extents` returns 1-based cell indices
- [ ] `getcell` returns polygon with correct vertex type
- [ ] `best_manifold` returns correct manifold type
