#=
# Interfaces for trees

This file contains interfaces for tree types, mainly quadtrees for now. 

## AbstractCurvilinearGrid

AbstractCurvilinearGrid is the abstract supertype for all quadtree bases.
Here, we define a "Curvilinear grid" as a grid of polygons that could be represented as a 2D matrix, 
where all neighbours in space are also neighbours in the matrix.

Then, [`Trees.AbstractQuadtreeCursor`](@ref) defines how you go down that quadtree.

Implementations of `AbstractCurvilinearGrid` are [`Trees.RegularGrid`](@ref), [`Trees.ExplicitPolygonGrid`](@ref), and [`Trees.CellBasedGrid`](@ref).
=#

"""
    abstract type AbstractCurvilinearGrid

Abstract supertype for all quadtree types.
The type itself should store the representation of the "base" of the quadtree,
which should fit into the `QuadtreeCursor` type.

The `QuadtreeCursor` type is a cursor that can be used to traverse the quadtree.
It should be able to traverse the quadtree in a depth-first manner, and should be able to
get the child nodes of the current node.

Since the quadtree structure is the same, you would broadly need to provide:

```julia
getcell(quadtree, i, j) -> GI.Polygon
ncells(quadtree, dim::Int) -> Int
cell_range_extent(quadtree, irange::UnitRange{Int}, jrange::UnitRange{Int})
```
, i.e., provide an implementation for [`Trees.getcell`](@ref), [`Trees.ncells`](@ref), and [`Trees.cell_range_extent`](@ref).

and then you may also want to specialize on `STI.node_extent(::QuadtreeCursor{<: YourQuadtreeType}) -> GO.UnitSpherical.SphericalCap{Float64}`
"""
abstract type AbstractCurvilinearGrid end

"""
    getcell(quadtree::AbstractCurvilinearGrid, i::Int, j::Int) -> GI.Polygon
    getcell(quadtree::AbstractCurvilinearGrid, idx::Integer) -> GI.Polygon
    getcell(quadtree::AbstractCurvilinearGrid, idx::CartesianIndex{2}) -> GI.Polygon

Get the cell at the given indices from the underlying quadtree object.  

If implementing a [`Trees.AbstractCurvilinearGrid`](@ref), you should implement `getcell(quadtree, i, j)`.
Other implementations are built on top of this and [`Trees.ncells`](@ref).
"""
function getcell(quadtree::AbstractCurvilinearGrid, i::Int, j::Int)
    error("getcell not implemented for $(typeof(quadtree))")
end

"""
    ncells(quadtree::AbstractCurvilinearGrid, dim::Int) -> Int
    ncells(quadtree::AbstractCurvilinearGrid) -> (Int, Int)

Get the number of cells in the given dimension of the underlying quadtree object.
This is used to determine the size of the quadtree in the given dimension.

If implementing a [`Trees.AbstractCurvilinearGrid`](@ref), you should implement `ncells(quadtree, dim)`.
Other implementations are built on top of this basic method.
"""
function ncells(quadtree::AbstractCurvilinearGrid, dim::Int)
    error("ncells not implemented for $(typeof(quadtree))")
end

"""
    cell_range_extent(quadtree::AbstractCurvilinearGrid, irange::UnitRange{Int}, jrange::UnitRange{Int}) -> GO.UnitSpherical.SphericalCap{Float64}

Get the extent of the cells in the given range of indices.
"""
function cell_range_extent(quadtree::AbstractCurvilinearGrid, irange::UnitRange{Int}, jrange::UnitRange{Int})
    error("cell_range_extent not implemented for $(typeof(quadtree))")
end

# ### Generic higher-level implementations

# Toplevel generic method to get all cells
function getcell(quadtree::AbstractCurvilinearGrid)
    return (getcell(quadtree, i, j) for i in 1:ncells(quadtree, 1), j in 1:ncells(quadtree, 2))
end
getcell(quadtree::AbstractCurvilinearGrid, idx::CartesianIndex{2}) = getcell(quadtree, idx[1], idx[2])
# Method to get cell from linear index
function getcell(quadtree::AbstractCurvilinearGrid, idx::Integer)
    ij = linear_to_cartesian_idx(quadtree, idx)
    return getcell(quadtree, ij.I...)
end

function linear_to_cartesian_idx(quadtree::AbstractCurvilinearGrid, idx::Integer)
    j, i = fldmod1(idx, ncells(quadtree, 1))
    return CartesianIndex(i, j)
end

function cartesian_to_linear_idx(quadtree::AbstractCurvilinearGrid, idx::CartesianIndex{2})
    return idx[1] + (idx[2] - 1) * ncells(quadtree, 1)
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

