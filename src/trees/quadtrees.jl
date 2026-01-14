#=

=#
import GeoInterface as GI, GeometryOps as GO
import GeometryOps: SpatialTreeInterface as STI
import StaticArrays


"""
    ExplicitPolygonGrid(polygons::AbstractMatrix)

A grid that is built from a matrix of pre-computed polygons.  This is the most explicit method with the least optimizations, but it is the most flexible.
"""
struct ExplicitPolygonGrid{PolyMatrixType <: AbstractMatrix} <: AbstractCurvilinearGrid
    polygons::PolyMatrixType
end

getcell(quadtree::ExplicitPolygonGrid, i::Int, j::Int) = quadtree.polygons[i, j]
ncells(quadtree::ExplicitPolygonGrid, dim::Int) = size(quadtree.polygons, dim)


"""
    CellBasedGrid(points::AbstractMatrix)

A grid that is built from a matrix of corner points.  This is more optimized than [`ExplicitPolygonGrid`](@ref)
because it knows the corner points of each polygon.
"""
struct CellBasedGrid{PointMatrixType <: AbstractMatrix} <: AbstractCurvilinearGrid
    points::PointMatrixType
end   

Base.@propagate_inbounds function getcell(quadtree::CellBasedGrid, i::Int, j::Int)
    @boundscheck begin
        if i < 1 || i >= size(quadtree.points, 1) || j < 1 || j >= size(quadtree.points, 2)
            error("Invalid index for cell based grid; got ($i, $j), but the matrix has $(size(quadtree.points) .- 1) polygons (for that .+1 points).")
        end
    end
    return GI.Polygon(StaticArrays.SA[GI.LinearRing(StaticArrays.SA[
        quadtree.points[i, j], 
        quadtree.points[i+1, j],
        quadtree.points[i+1, j+1],
        quadtree.points[i, j+1],
        quadtree.points[i, j]
    ])])
end
ncells(quadtree::CellBasedGrid, dim::Int) = size(quadtree.points, dim) - 1

"""
    RegularGrid(x::AbstractVector, y::AbstractVector)

A grid that is built from a regular grid of x and y coordinates.

This is optimized for regular grids but requires unit-spherical transforms
to be run on each call to `node_extent` - so it might be better to use a
[`CellBasedGrid`](@ref) instead if performance is a concern.
"""
struct RegularGrid{DX, DY} <: AbstractCurvilinearGrid
    x::DX
    y::DY
end

ncells(quadtree::RegularGrid, dim::Int) = length(dim == 1 ? quadtree.x : quadtree.y) - 1
function getcell(quadtree::RegularGrid, i::Int, j::Int)
    @boundscheck begin
        if i < 1 || i > ncells(quadtree, 1) || j < 1 || j > ncells(quadtree, 2)
            error("Invalid index for regular grid; got ($i, $j), but the grid has $(ncells(quadtree, 1)) cells in the x direction and $(ncells(quadtree, 2)) cells in the y direction.")
        end
    end
    return GI.Polygon(StaticArrays.SA[GI.LinearRing(StaticArrays.SA[
        (quadtree.x[i], quadtree.y[j]), (quadtree.x[i+1], quadtree.y[j]), (quadtree.x[i+1], quadtree.y[j+1]), (quadtree.x[i], quadtree.y[j+1]), (quadtree.x[i], quadtree.y[j])
    ])])
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
    return GO.UnitSpherical.SphericalCap(center, distance*1.0001)
end
function cell_range_extent(q::CellBasedGrid, irange::UnitRange{Int}, jrange::UnitRange{Int})
    imin, imax = extrema(irange)
    jmin, jmax = extrema(jrange)
    # For cell based we need 1 to the max indices from cell space 
    # to get the max indices in point space (`(n,m) -> (n+1, m+1)`)
    imax += 1
    jmax += 1

    quadtree_points = q.points
    corner_points = (quadtree_points[imin, jmin], quadtree_points[imin, jmax], quadtree_points[imax, jmin], quadtree_points[imax, jmax])

    # Collect points from all border polygons
    other_points = typeof(GI.getpoint(GI.getexterior(getcell(q, imin, jmin)), 1))[]
    sizehint!(other_points, (imax - imin + 1) * (jmax - jmin + 1))
    # Top and bottom rows (all columns)
    append!(other_points, view(q.points, imin, jmin:jmax))
    if imax != imin
        append!(other_points, view(q.points, imax, jmin:jmax))
    end
    if jmax != jmin
        append!(other_points, view(q.points, imin:imax, jmax))
    end
    return circle_from_four_corners(corner_points, other_points)
end

function cell_range_extent(q::RegularGrid, irange::UnitRange{Int}, jrange::UnitRange{Int})
    imin, imax = extrema(irange)
    jmin, jmax = extrema(jrange)
    # For cell based we need 1 to the max indices from cell space 
    # to get the max indices in point space (`(n,m) -> (n+1, m+1)`)
    imax += 1
    jmax += 1

    corner_points = ((q.x[imin], q.y[jmin]), (q.x[imin], q.y[jmax]), (q.x[imax], q.y[jmin]), (q.x[imax], q.y[jmax]))

    # Collect points from all border polygons
    other_points = typeof(GI.getpoint(GI.getexterior(getcell(q, imin, jmin)), 1))[]
    sizehint!(other_points, (imax - imin + 1) * (jmax - jmin + 1))
    # Top and bottom rows (all columns)
    append!(other_points, tuple.(q.x[imin], q.y[jmin:jmax]))
    if imax != imin
        append!(other_points, tuple.(q.x[imax], q.y[jmin:jmax]))
    end
    if jmax != jmin
        append!(other_points, tuple.(q.x[imin:imax], q.y[jmax]))
    end
    return circle_from_four_corners(corner_points, other_points)
end