#=
# Interfaces for trees

This file contains interfaces for tree types, mainly quadtrees for now. 

## Treeify

[`Trees.treeify`](@ref) is a function that should take a manifold and a grid specification (that can be any struct from anywhere)
and return a `SpatialTreeInterface`-compliant tree.

This could mean that you take the vertices of a grid, put them in an `AbstractCurvilinearGrid`,
and then wrap those in a `QuadtreeCursor`.  But you could do anything you want - so long as the 
thing that is returned implements the `SpatialTreeInterface` methods.
=#
import GeometryOpsCore as GOCore
import GeometryOps as GO
import GeometryOps: SpatialTreeInterface as STI
import ConstructionBase
import Extents
import SortTileRecursiveTree # in order to implement the `getcell/ncell` interface

"""
    should_parallelize(node, extent) -> Bool

!!! warning "Deprecated"
    The multithreaded dual-tree traversal no longer consults this function - it
    sizes its tasks from [`Trees.split_weight`](@ref) instead.  Kept so existing
    method definitions keep loading; slated for removal in a breaking release.

Formerly: decide whether to spawn a parallel task at `node`, whose bounding
region is `extent`, during the multithreaded dual-tree traversal.
"""
function should_parallelize end

# Nothing in the package calls this any more; the fallback is only here to warn.
function should_parallelize(node, extent)
    Base.depwarn("`Trees.should_parallelize` is no longer consulted by the multithreaded traversal - define `Trees.split_weight` instead.", :should_parallelize)
    return false
end

"""
    split_weight(node) -> Int

A fast estimate of how many leaves live under `node`.

This is used to build a multithreading plan in `multithreaded_dual_query`.
A wrong answer here will impact runtime, but not correctness.

By default, this has a fallback that tries to check `Trees.ncells(node)`,
and returns `1` if that is not applicable for the node.
"""
function split_weight(node::N) where N
    applicable(ncells, node) || return 1
    n = ncells(node)
    return n isa Tuple ? Int(prod(n)) : Int(n)
end

"""
    cell_index_count(tree) -> Int

The size of the dense, one-based cell-index domain emitted by `tree`.

This differs from [`Trees.ncells`](@ref) for a tree restricted to part of a
larger grid: `ncells` describes the cells below that node, while
`cell_index_count` describes the global index domain required to store its leaf
indices.  Trees whose leaf indices are local may use the fallback, which derives
the count from `ncells(tree)`.
"""
function cell_index_count(tree)
    n = ncells(tree)
    return n isa Tuple ? Int(prod(n)) : Int(n)
end

# Generic method to treeify "anything"
function treeify(manifold, grid)
    if STI.isspatialtree(grid)
        if applicable(getcell, grid, 1)
            return grid
        else
            error("grid is a SpatialTreeInterface-compliant tree, but does not implement `ConservativeRegridding.Trees.getcell` - please implement this method!")
        end
    elseif grid isa AbstractMatrix
        if GI.trait(first(grid)) isa GI.AbstractPolygonTrait
            return TopDownQuadtreeCursor(ExplicitPolygonGrid(manifold, grid))
        elseif GI.trait(first(grid)) isa GI.AbstractPointTrait
            return TopDownQuadtreeCursor(CellBasedGrid(manifold, grid))
        else
            error("grid is a matrix, but no element is a polygon or point - please implement `ConservativeRegridding.Trees.treeify` for this type!")
        end
    elseif Base.isiterable(typeof(grid))
        if all(g -> GI.trait(g) isa Union{GI.AbstractPolygonTrait, GI.AbstractMultiPolygonTrait}, grid)
            return STI.FlatNoTree(grid)
        else
            if GI.isgeometry(first(grid))
                error("grid is a iterable of geometries, but not all geometries are polygons - please implement `ConservativeRegridding.Trees.treeify` for this type!")
            else
                error("grid is a iterable, but no GeoInterface-compatible geometries were detected - please implement `ConservativeRegridding.Trees.treeify` for this type!")
            end
        end
    end
end

treeify(grid) = treeify(GOCore.best_manifold(grid), grid)

# Some example implementations
GOCore.best_manifold(grid::AbstractMatrix{<: GO.UnitSpherical.UnitSphericalPoint}) = GO.Spherical()
treeify(manifold::GO.Spherical, grid::AbstractMatrix{<: GO.UnitSpherical.UnitSphericalPoint}) = TopDownQuadtreeCursor(CellBasedGrid(manifold, grid))

GOCore.best_manifold(grid::NTuple{2, <: AbstractVector{<: Real}}) = GO.Planar()
treeify(manifold::GOCore.Manifold, grid::NTuple{2, <: AbstractVector{<: Real}}) = TopDownQuadtreeCursor(RegularGrid(manifold, grid...))

#=
## AbstractCurvilinearGrid

AbstractCurvilinearGrid is the abstract supertype for all quadtree bases.
Here, we define a "Curvilinear grid" as a grid of polygons that could be represented as a 2D matrix, 
where all neighbours in space are also neighbours in the matrix.

Then, [`Trees.AbstractQuadtreeCursor`](@ref) defines how you go down that quadtree.

Implementations of `AbstractCurvilinearGrid` are [`Trees.RegularGrid`](@ref), [`Trees.ExplicitPolygonGrid`](@ref), and [`Trees.CellBasedGrid`](@ref).
=#

"""
    abstract type AbstractCurvilinearGrid{M <: GOCore.Manifold}

Abstract supertype for all curvilinear grid types, parameterized by the manifold `M`
(e.g. `Planar`, `Spherical`) the grid lives on, so algorithms can dispatch on the
manifold — see the generic spherical [`Trees.cell_range_extent`](@ref).
The type itself should store the representation of the "base" of the quadtree,
which should fit into the `QuadtreeCursor` type.

The `QuadtreeCursor` type is a cursor that can be used to traverse the quadtree.
It should be able to traverse the quadtree in a depth-first manner, and should be able to
get the child nodes of the current node.

Since the quadtree structure is the same, you would broadly need to provide:

```julia
getcell(grid, i, j) -> GI.Polygon
ncells(grid, dim::Int) -> Int
cell_range_extent(grid, irange::UnitRange{Int}, jrange::UnitRange{Int})
```
, i.e., provide an implementation for [`Trees.getcell`](@ref), [`Trees.ncells`](@ref), and [`Trees.cell_range_extent`](@ref).

and then you may also want to specialize on `STI.node_extent(::QuadtreeCursor{<: YourQuadtreeType}) -> GO.UnitSpherical.SphericalCap{Float64}`
"""
abstract type AbstractCurvilinearGrid{M <: GOCore.Manifold} end

# A grid carries the manifold it was built on, so there is nothing to guess.
GOCore.best_manifold(grid::AbstractCurvilinearGrid) = GOCore.manifold(grid)

"""
    treeify(manifold::GOCore.Manifold, grid::AbstractCurvilinearGrid)

Wrap `grid` in a [`Trees.TopDownQuadtreeCursor`](@ref), overriding the grid's own manifold
with `manifold` if they differ.  The override rebuilds the grid via `ConstructionBase`,
sharing its geometry rather than copying, since [`cell_range_extent`](@ref) dispatches on
the grid's own manifold type.
"""
treeify(manifold::GOCore.Manifold, grid::AbstractCurvilinearGrid) = TopDownQuadtreeCursor(
    GOCore.manifold(grid) === manifold ? grid : ConstructionBase.setproperties(grid, (; manifold))
)

"""
    getcell(grid::AbstractCurvilinearGrid, i::Int, j::Int) -> GI.Polygon
    getcell(grid::AbstractCurvilinearGrid, idx::Integer) -> GI.Polygon
    getcell(grid::AbstractCurvilinearGrid, idx::CartesianIndex{2}) -> GI.Polygon

Get the cell at the given indices from the underlying grid object.  

If implementing a [`Trees.AbstractCurvilinearGrid`](@ref), you should implement `getcell(grid, i, j)`.
Other implementations are built on top of this and [`Trees.ncells`](@ref).
"""
function getcell(grid::AbstractCurvilinearGrid, i::Int, j::Int)
    error("getcell not implemented for $(typeof(grid))")
end

"""
    ncells(grid::AbstractCurvilinearGrid, dim::Int) -> Int
    ncells(grid::AbstractCurvilinearGrid) -> (Int, Int)

Get the number of cells in the given dimension of the underlying grid object.
This is used to determine the size of the grid in the given dimension.

If implementing a [`Trees.AbstractCurvilinearGrid`](@ref), you should implement `ncells(grid, dim)`.
Other implementations are built on top of this basic method.
"""
function ncells(grid::AbstractCurvilinearGrid, dim::Int)
    error("ncells not implemented for $(typeof(grid))")
end

cell_index_count(grid::AbstractCurvilinearGrid) =
    Int(ncells(grid, 1)) * Int(ncells(grid, 2))

"""
    cell_range_extent(grid::AbstractCurvilinearGrid, irange::UnitRange{Int}, jrange::UnitRange{Int}) -> GO.UnitSpherical.SphericalCap{Float64}

Get the extent of the cells in the given range of indices.
"""
function cell_range_extent(grid::AbstractCurvilinearGrid, irange::UnitRange{Int}, jrange::UnitRange{Int})
    error("cell_range_extent not implemented for $(typeof(grid))")
end

"""
    getvertex(grid::AbstractCurvilinearGrid, i::Int, j::Int)

Get the grid vertex at *point-index* `(i, j)`. Point indices run `1:(n+1) × 1:(m+1)`
for an `n × m` cell grid — one more than the cell count per dimension, since adjacent
cells share corners. For spherical grids this returns a
`GO.UnitSpherical.UnitSphericalPoint`.

Part of the [`Trees.AbstractCurvilinearGrid`](@ref) interface (sibling to
[`Trees.getcell`](@ref)), used by the spherical [`Trees.cell_range_extent`](@ref) to
build bounding caps from the corners and `CurvilinearGridPerimeterPoints` of an index
range. Structured grids on the sphere should implement it; planar grids need not.
"""
function getvertex(grid::AbstractCurvilinearGrid, i::Int, j::Int)
    error("getvertex not implemented for $(typeof(grid))")
end

# ### Generic higher-level implementations

# Toplevel generic method to get all cells
function getcell(grid::AbstractCurvilinearGrid)
    return (getcell(grid, i, j) for i in 1:ncells(grid, 1), j in 1:ncells(grid, 2))
end
getcell(grid::AbstractCurvilinearGrid, idx::CartesianIndex{2}) = getcell(grid, idx[1], idx[2])
# Method to get cell from linear index
function getcell(grid::AbstractCurvilinearGrid, idx::Integer)
    ij = linear_to_cartesian_idx(grid, idx)
    return getcell(grid, ij.I...)
end

function linear_to_cartesian_idx(grid::AbstractCurvilinearGrid, idx::Integer)
    j, i = fldmod1(idx, ncells(grid, 1))
    return CartesianIndex(i, j)
end

function cartesian_to_linear_idx(grid::AbstractCurvilinearGrid, idx::CartesianIndex{2})
    return idx[1] + (idx[2] - 1) * ncells(grid, 1)
end

#=
## AbstractQuadtreeCursor

AbstractQuadtreeCursor is the abstract supertype for all quadtree cursor types.
This is the type that you use to traverse the quadtree.

Implementations of `AbstractQuadtreeCursor` are [`Trees.QuadtreeCursor`](@ref) and [`Trees.TopDownQuadtreeCursor`](@ref).
=#

"""
    abstract type AbstractQuadtreeCursor end

Abstract supertype for all quadtree cursor types.
This is the type that you use to traverse the quadtree
defined on some `AbstractCurvilinearGrid`.

Subtypes of `AbstractQuadtreeCursor` should implement the following methods:
- `getgrid(cursor::AbstractQuadtreeCursor) -> AbstractCurvilinearGrid`
"""
abstract type AbstractQuadtreeCursor end

function getgrid(cursor::AbstractQuadtreeCursor)
    error("getgrid not implemented for $(typeof(cursor))")
end

# ### Generic implementations
STI.isspatialtree(::Type{<: AbstractQuadtreeCursor}) = true
STI.nchild(cursor::AbstractQuadtreeCursor) = error("GO.STI.nchild not implemented for $(typeof(cursor))")
STI.getchild(cursor::AbstractQuadtreeCursor, i::Int) = error("GO.STI.getchild not implemented for $(typeof(cursor))")
STI.isleaf(cursor::AbstractQuadtreeCursor) = error("GO.STI.isleaf not implemented for $(typeof(cursor))")
STI.child_indices_extents(cursor::AbstractQuadtreeCursor) = error("GO.STI.child_indices_extents not implemented for $(typeof(cursor))")
STI.node_extent(cursor::AbstractQuadtreeCursor) = error("GO.STI.node_extent not implemented for $(typeof(cursor))")

"""
    extent_is_expensive(grid) -> Bool

Whether `cell_range_extent(grid, irange, jrange)` *computes* the range's bounding extent
rather than reading it off a couple of coordinate vectors.  True for every grid but the
planar regular one, where the answer is four array reads.

This is the grid-level fact behind `STI.node_extent_is_expensive` for the quadtree
cursors: a cursor's `node_extent` is exactly `cell_range_extent` over its leaf range, so
it is expensive precisely when the grid's is.  The search in
[`ConservativeRegridding.CachedDualDepthFirstSearch`](@ref) reads the cursor-level trait
to decide whether caching a node's child extents pays for itself.

Defined on the type, so it stays a compile-time constant; the instance method forwards.
"""
extent_is_expensive(grid) = extent_is_expensive(typeof(grid))
extent_is_expensive(::Type{<: AbstractCurvilinearGrid}) = true


# ## Implementations for external types
function ncells(tree::STI.FlatNoTree)
    return length(tree.geometries)
end
function getcell(tree::STI.FlatNoTree, idx::Integer)
    return tree.geometries[idx]
end
function getcell(tree::STI.FlatNoTree)
    return tree.geometries
end

function ncells(tree::SortTileRecursiveTree.STRtree)
    n = 0
    STI.depth_first_search(e -> true, tree) do i
        n += 1
    end
    return n
end
