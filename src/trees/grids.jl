#=
# Grids

This file contains definitions for various expressions of curvilinear grids.

They range from the most explicit (a matrix of polygons in [`Trees.ExplicitPolygonGrid`](@ref)),
to a matrix of corner vertices (in [`Trees.CellBasedGrid`](@ref)), to a regular grid (in [`Trees.RegularGrid`](@ref)).

These all subtype [`Trees.AbstractCurvilinearGrid`](@ref), which defines the interface for all curvilinear grids.
See `interfaces.jl` in this directory for more information there.

The idea behind all of these grids is that you can then build spatial trees from the grids,
but specialize the implementation (specifically `STI.node_extent`) for each grid type to get
maximum performance and accuracy.
=#
import GeoInterface as GI, GeometryOps as GO
import GeometryOpsCore as GOCore

import GeometryOps: SpatialTreeInterface as STI
import Extents

import StaticArrays

#=
## Definitions

Here we'll define the concrete implementations of each grid type.
The `cell_range_extent` methods will come later since they are specific
to the manifold the grid lives on.
=#

"""
    ExplicitPolygonGrid(polygons::AbstractMatrix)

A grid that is built from a matrix of pre-computed polygons.  This is the most explicit method with the least optimizations, but it is the most flexible.
"""
struct ExplicitPolygonGrid{M <: GOCore.Manifold, PolyMatrixType <: AbstractMatrix} <: AbstractCurvilinearGrid
    manifold::M
    polygons::PolyMatrixType
end
GOCore.manifold(grid::ExplicitPolygonGrid) = grid.manifold

getcell(grid::ExplicitPolygonGrid, i::Int, j::Int) = grid.polygons[i, j]
ncells(grid::ExplicitPolygonGrid, dim::Int) = size(grid.polygons, dim)


"""
    CellBasedGrid(points::AbstractMatrix)

A grid that is built from a matrix of corner points.  This is more optimized than [`ExplicitPolygonGrid`](@ref)
because it knows the corner points of each polygon.
"""
struct CellBasedGrid{M <: GOCore.Manifold, PointMatrixType <: AbstractMatrix} <: AbstractCurvilinearGrid
    manifold::M
    points::PointMatrixType
end
function CellBasedGrid(points::AbstractMatrix)
    if eltype(points) <: GO.UnitSpherical.UnitSphericalPoint
        CellBasedGrid(GO.Spherical(), points)
    else
        CellBasedGrid(GO.Planar(), points)
    end
end

GOCore.manifold(grid::CellBasedGrid) = grid.manifold

Base.@propagate_inbounds function getcell(grid::CellBasedGrid, i::Int, j::Int)
    @boundscheck begin
        if i < 1 || i >= size(grid.points, 1) || j < 1 || j >= size(grid.points, 2)
            error("Invalid index for cell based grid; got ($i, $j), but the matrix has $(size(grid.points) .- 1) polygons (for that .+1 points).")
        end
    end
    return GI.Polygon(StaticArrays.SA[GI.LinearRing(StaticArrays.SA[
        grid.points[i, j], 
        grid.points[i+1, j],
        grid.points[i+1, j+1],
        grid.points[i, j+1],
        grid.points[i, j]
    ])])
end
ncells(grid::CellBasedGrid, dim::Int) = size(grid.points, dim) - 1

"""
    RegularGrid(x::AbstractVector, y::AbstractVector)

A grid that is built from a regular grid of x and y coordinates.

This is optimized for regular grids but requires unit-spherical transforms
to be run on each call to `node_extent` - so it might be better to use a
[`CellBasedGrid`](@ref) instead if performance is a concern.
"""
struct RegularGrid{M <: GOCore.Manifold, DX, DY} <: AbstractCurvilinearGrid
    manifold::M
    x::DX
    y::DY
end
RegularGrid(x, y) = RegularGrid(GO.Planar(), x, y)

GOCore.manifold(grid::RegularGrid) = grid.manifold

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


#=
## `cell_range_extent` methods

These are specialized for each grid type and each manifold.

### `Planar` manifolds

On `Planar` manifolds the implementation is always simple - just
compute the `Extents.Extent` and return it.
=#

function cell_range_extent(q::ExplicitPolygonGrid{<: GO.Planar}, irange::UnitRange{Int}, jrange::UnitRange{Int})
    return mapreduce(Extents.union, GI.extent, (getcell(q, i, j) for i in irange, j in jrange))
end

function cell_range_extent(q::CellBasedGrid{<: GO.Planar}, irange::UnitRange{Int}, jrange::UnitRange{Int})
    xrange = extrema(p -> GI.x(p), view(q.points, first(irange):last(irange)+1, first(jrange):last(jrange)+1))
    yrange = extrema(p -> GI.y(p), view(q.points, first(irange):last(irange)+1, first(jrange):last(jrange)+1))
    return Extents.Extent(; X=xrange, Y=yrange)
end

function cell_range_extent(q::RegularGrid{<: GO.Planar}, irange::UnitRange{Int}, jrange::UnitRange{Int})
    xrange = (q.x[first(irange)], q.x[last(irange)+1])
    yrange = (q.y[first(jrange)], q.y[last(jrange)+1])
    return Extents.Extent(; X=xrange, Y=yrange)
end

#=
### `Spherical` manifolds

On `Spherical` manifolds the implementation is more complex - we need to compute the
spherical cap that encloses all the points in the given range.

Here you will notice that we've only defined methods for the structured grids - 
for the `ExplicitPolygonGrid` case, it still needs to be implemented (TODO).

The way we do this here is that we compute the circle that encloses the four corners
of the given range, and then expand it such that it encloses all other points along the borders.

Since we know our grids are curvilinear, this is broadly okay.  It's not perfectly efficient, though.
=#

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

function cell_range_extent(q::CellBasedGrid{<: GO.Spherical}, irange::UnitRange{Int}, jrange::UnitRange{Int})
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

function cell_range_extent(q::RegularGrid{<: GO.Spherical}, irange::UnitRange{Int}, jrange::UnitRange{Int})
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