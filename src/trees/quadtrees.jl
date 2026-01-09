import GeoInterface as GI, GeometryOps as GO
import GeometryOps: SpatialTreeInterface as STI

"""
    abstract type AbstractQuadtree

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
```

and then you may also want to specialize on `STI.node_extent(::QuadtreeCursor{<: YourQuadtreeType}) -> GO.UnitSpherical.SphericalCap{Float64}`
"""
abstract type AbstractQuadtree end

"""
    getcell(quadtree::AbstractQuadtree, i::Int, j::Int) -> GI.Polygon

Get the cell at the given indices from the underlying quadtree object.
"""
function getcell(quadtree::AbstractQuadtree, i::Int, j::Int)
    error("getcell not implemented for $(typeof(quadtree))")
end

"""
    ncells(quadtree::AbstractQuadtree, dim::Int) -> Int

Get the number of cells in the given dimension of the underlying quadtree object.

This is used to determine the size of the quadtree in the given dimension.
"""
function ncells(quadtree::AbstractQuadtree, dim::Int)
    error("ncells not implemented for $(typeof(quadtree))")
end
# Toplevel generic method to get all cells
function getcell(quadtree::AbstractQuadtree)
    return (getcell(quadtree, i, j) for i in 1:ncells(quadtree, 1), j in 1:ncells(quadtree, 2))
end
getcell(quadtree::AbstractQuadtree, idx::CartesianIndex{2}) = getcell(quadtree, idx[1], idx[2])

"""
    ExplicitPolygonQuadtree(polygons::AbstractMatrix)

A quadtree that is built from a matrix of pre-computed polygons.  This is the most explicit method with the least optimizations, but it is the most flexible.
"""
struct ExplicitPolygonQuadtree{PolyMatrixType <: AbstractMatrix} <: AbstractQuadtree
    polygons::PolyMatrixType
end

getcell(quadtree::ExplicitPolygonQuadtree, i::Int, j::Int) = quadtree.polygons[i, j]
ncells(quadtree::ExplicitPolygonQuadtree, dim::Int) = size(quadtree.polygons, dim)

"""
    CellBasedQuadtree(points::AbstractMatrix)

A quadtree that is built from a matrix of corner points.  This is more optimized than [`ExplicitPolygonQuadtree`](@ref) 
because it knows the corner points of each polygon.
"""
struct CellBasedQuadtree{PointMatrixType <: AbstractMatrix} <: AbstractQuadtree
    points::PointMatrixType
end   

Base.@propagate_inbounds function getcell(quadtree::CellBasedQuadtree, i::Int, j::Int)
    @boundscheck begin
        if i < 1 || i >= size(quadtree.points, 1) || j < 1 || j >= size(quadtree.points, 2)
            error("Invalid index for cell based quadtree; got ($i, $j), but the matrix has $(size(quadtree.points) .- 1) polygons (for that .+1 points).")
        end
    end
    return GI.Polygon([GI.LinearRing([
        quadtree.points[i, j], 
        quadtree.points[i, j+1],
        quadtree.points[i+1, j+1],
        quadtree.points[i+1, j],
        quadtree.points[i, j]
    ])])
end
ncells(quadtree::CellBasedQuadtree, dim::Int) = size(quadtree.points, dim) - 1

"""
    RegularGridQuadtree(x::AbstractVector, y::AbstractVector)

A quadtree that is built from a regular grid of x and y coordinates.  

This is optimized for regular grids but requires unit-spherical transforms
to be run on each call to `node_extent` - so it might be better to use a 
[`CellBasedQuadtree`](@ref) instead if performance is a concern.
"""
struct RegularGridQuadtree{DX, DY} <: AbstractQuadtree
    x::DX
    y::DY
end

ncells(quadtree::RegularGridQuadtree, dim::Int) = length(dim == 1 ? quadtree.x : quadtree.y) - 1
function getcell(quadtree::RegularGridQuadtree, i::Int, j::Int)
    @boundscheck begin
        if i < 1 || i > ncells(quadtree, 1) || j < 1 || j > ncells(quadtree, 2)
            error("Invalid index for regular grid quadtree; got ($i, $j), but the grid has $(ncells(quadtree, 1)) cells in the x direction and $(ncells(quadtree, 2)) cells in the y direction.")
        end
    end
    return GI.Polygon([GI.LinearRing([
        (quadtree.x[i], quadtree.y[j]), (quadtree.x[i+1], quadtree.y[j]), (quadtree.x[i+1], quadtree.y[j+1]), (quadtree.x[i], quadtree.y[j+1]), (quadtree.x[i], quadtree.y[j])
    ])])
end

"""
    QuadtreeCursor(quadtree::AbstractQuadtree)

A cursor for traversing the quadtree.  This is used to traverse the quadtree in a depth-first manner, and to get the child nodes of the current node.

## Fields

$(DocStringExtensions.FIELDS)
"""
struct QuadtreeCursor{QuadtreeType <: AbstractQuadtree}
    "The leaf level representation that the quadtree is built on top of."
    quadtree::QuadtreeType
    "The index of the current cell at the current level in the quadtree."
    idx::CartesianIndex{2}
    "The level of the cursor - 1 is the base i.e. smallest polygon level, as you increase the number you increase the size of the thing."
    level::Int
end

function QuadtreeCursor(quadtree::AbstractQuadtree)
    # Need 2^(level-1) >= max_size, so level >= log2(max_size) + 1
    max_level = ceil(Int, log2(max(ncells(quadtree, 1), ncells(quadtree, 2)))) + 1
    return QuadtreeCursor(quadtree, CartesianIndex(1, 1), max_level)
end

# ### General methods for any quadtree type

function leaf_idxs(q::QuadtreeCursor)
     # Calculate the range of leaf indices covered by this node
     scale = 2^(q.level - 1)
     psize = (ncells(q.quadtree, 1), ncells(q.quadtree, 2))
     
     # Compute and clamp all indices to polygon matrix bounds
     imin = min((q.idx[1] - 1) * scale + 1, psize[1])
     imax = min(q.idx[1] * scale, psize[1])
     jmin = min((q.idx[2] - 1) * scale + 1, psize[2])
     jmax = min(q.idx[2] * scale, psize[2])

     return imin:imax, jmin:jmax
end

function getcell(q::QuadtreeCursor)
    is, js = leaf_idxs(q)
    return (getcell(q.quadtree, i, j) for i in is, j in js)
end

STI.isspatialtree(::Type{<: QuadtreeCursor}) = true
function STI.nchild(q::QuadtreeCursor)
    imax, jmax = (q.idx.I .- 1) .* (2^q.level) .+ 1
    ioff = ncells(q.quadtree, 1) - imax
    joff = ncells(q.quadtree, 2) - jmax
    nchildren = if ioff > 1
        2
    elseif ioff == 1
        1
    elseif ioff < 1
        0
    end * if joff > 1
        2
    elseif joff == 1
        1
    elseif joff < 1
        0
    end

    return nchildren
end

function STI.isleaf(q::QuadtreeCursor)
    q.level < 1 && throw(ArgumentError("Quadtree level must be greater than 1; got $(q.level).  Something went wrong!"))
    return q.level == 1
end

function STI.getchild(q::QuadtreeCursor, i::Int)
    i > STI.nchild(q) && throw(ArgumentError("Invalid child index; got $i, but there are only $(STI.nchild(q)) children in the node."))
    new_idx = ((q.idx.I .- 1) .* 2) .+ CartesianIndices((1:2, 1:2))[i].I
    return QuadtreeCursor(q.quadtree, CartesianIndex(new_idx), q.level - 1)
end


# ### Specialized methods for each quadtree type
# #### Node extent for explicit polygon quadtree
# TOOD port back from old implmementation
# #### Node extent for cell based quadtree

using LinearAlgebra
function circle_from_four_corners(corner_points, other_points)
    p1, p2, p3, p4 = GO.UnitSphereFromGeographic().(corner_points)
    center = LinearAlgebra.normalize((p1 .+ p2 .+ p3 .+ p4) ./ 4)
    # Midpoints of lines / edges
    p12 = GO.UnitSpherical.slerp(p1, p2, 0.5)
    p23 = GO.UnitSpherical.slerp(p2, p3, 0.5)
    p34 = GO.UnitSpherical.slerp(p3, p4, 0.5)
    p41 = GO.UnitSpherical.slerp(p4, p1, 0.5)
    # Distance to the furthest corner or edge midpoint
    corner_distance = maximum(p -> GO.spherical_distance(center, p), (p1, p2, p3, p4, p12, p23, p34, p41))
    # Distance to the furthest other point
    other_distance = maximum(p -> GO.spherical_distance(center, p), GO.UnitSphereFromGeographic().(other_points); init = corner_distance)
    # Distance to the furthest point
    distance = max(corner_distance, other_distance)
    #The `*1.0001` is done to not miss intersections through numerical inaccuracies
    return GO.UnitSpherical.SphericalCap(center, distance*1.0001, max_radius = distance*1.0001)
end
function _corner_and_other_points_for_circle_from_corners(q::QuadtreeCursor{<: CellBasedQuadtree})
    if STI.isleaf(q)
        pts = GI.getpoint.(([getcell(q.quadtree, q.idx[1], q.idx[2])],), 1:4)
        return (pts, pts)
    else
        # Calculate the range of leaf indices covered by this node
        scale = 2^(q.level - 1)
        psize = (ncells(q.quadtree, 1), ncells(q.quadtree, 2))
        
        # Compute and clamp all indices to polygon matrix bounds
        irange, jrange = leaf_idxs(q)
        imin, imax = extrema(irange)
        jmin, jmax = extrema(jrange)

        quadtree_points = q.quadtree.points
        corner_points = (quadtree_points[imin, jmin], quadtree_points[imin, jmax], quadtree_points[imax, jmin], quadtree_points[imax, jmax])

        # Collect points from all border polygons
        other_points = typeof(GI.getpoint(GI.getexterior(getcell(q.quadtree, imin, jmin)), 1))[]
        sizehint!(other_points, (imax - imin + 1) * (jmax - jmin + 1))
        # Top and bottom rows (all columns)
        append!(other_points, view(q.quadtree.points, imin, jmin:jmax))
        if imax != imin
            append!(other_points, view(q.quadtree.points, imax, jmin:jmax))
        end
        if jmax != jmin
            append!(other_points, view(q.quadtree.points, imin:imax, jmax))
        end
        
        return (corner_points, other_points)
    end
end

function STI.node_extent(q::QuadtreeCursor{<: CellBasedQuadtree})
    return circle_from_four_corners(_corner_and_other_points_for_circle_from_corners(q)...)
end

function STI.node_extent(q::QuadtreeCursor{<: RegularGridQuadtree})
    max_level = ceil(Int, log2(max(ncells(q.quadtree, 1), ncells(q.quadtree, 2)))) + 1
    if q.level == max_level
        qt = q.quadtree
        if qt.x[1] == -180 && qt.x[end] == 180 && qt.y[1] == -90 && qt.y[end] == 90
            return GO.UnitSpherical.SphericalCap(GO.UnitSphericalPoint((0.,0.,1.)), pi)
        end
    end
    # Compute and clamp all indices to polygon matrix bounds
    irange, jrange = leaf_idxs(q)
    imin, imax = extrema(irange)
    jmin, jmax = extrema(jrange)

    x, y = q.quadtree.x, q.quadtree.y

    return circle_from_four_corners(
        ((x[imin], y[jmin]), (x[imin], y[jmax]), (x[imax], y[jmin]), (x[imax], y[jmax])),
        ()
    )
end
