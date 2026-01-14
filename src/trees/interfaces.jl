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

function treeify(manifold, grid)
    error("treeify not implemented for $(typeof(grid))")
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
    abstract type AbstractCurvilinearGrid

Abstract supertype for all curvilinear grid types.
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
abstract type AbstractCurvilinearGrid end

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

"""
    cell_range_extent(grid::AbstractCurvilinearGrid, irange::UnitRange{Int}, jrange::UnitRange{Int}) -> GO.UnitSpherical.SphericalCap{Float64}

Get the extent of the cells in the given range of indices.
"""
function cell_range_extent(grid::AbstractCurvilinearGrid, irange::UnitRange{Int}, jrange::UnitRange{Int})
    error("cell_range_extent not implemented for $(typeof(grid))")
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
