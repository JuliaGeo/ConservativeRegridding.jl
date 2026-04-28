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

ExplicitPolygonGrid(polygons::AbstractMatrix) = ExplicitPolygonGrid(GO.Planar(), polygons)

getcell(grid::ExplicitPolygonGrid, i::Int, j::Int) = grid.polygons[i, j]
ncells(grid::ExplicitPolygonGrid, dim::Int) = size(grid.polygons, dim)


"""
    CellBasedGrid(points::AbstractMatrix)

A grid that is built from a matrix of corner points.  This is more optimized than [`ExplicitPolygonGrid`](@ref)
because it knows the corner points of each polygon.

For a cell based grid with n by m cells, the points matrix will have n+1 by m+1 points.
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
    return mapreduce(GI.extent, Extents.union, (getcell(q, i, j) for i in irange, j in jrange))
end

function cell_range_extent(q::ExplicitPolygonGrid{<: GO.Spherical}, irange::UnitRange{Int}, jrange::UnitRange{Int})
    # Collect all unique points from polygons in the range and compute a bounding SphericalCap.
    all_points = GO.UnitSpherical.UnitSphericalPoint[]
    for j in jrange, i in irange
        poly = getcell(q, i, j)
        for pt in GI.getpoint(GI.getexterior(poly))
            push!(all_points, pt)
        end
    end
    isempty(all_points) && return GO.UnitSpherical.SphericalCap(GO.UnitSpherical.UnitSphericalPoint(0.0, 0.0, 1.0), 0.0)
    center = LinearAlgebra.normalize(sum(all_points) / length(all_points))
    radius = maximum(p -> GO.spherical_distance(center, p), all_points)
    return GO.UnitSpherical.SphericalCap(center, radius * 1.0001)
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

#=
Spherical `cell_range_extent` machinery.

The bounding-cap algorithm needs the four corners of the index rectangle plus
points sampled along its perimeter (because cell edges follow constant-lat /
constant-lon, not great-circles, so they bulge differently). Earlier versions
materialized the perimeter into a fresh `Vector` per call — that drove most of
the GC pressure in the dual-DFS hot path.

Now the perimeter is a stack-resident iterator (`PerimeterPoints`) that yields
points on demand via a small per-grid `_pt_at` method, and the cap builder
consumes any iterator over `UnitSphericalPoint`s. No buffers, no shared state,
threadsafe by construction.
=#

# Per-grid: materialize the UnitSphericalPoint at point-index (i, j).
@inline _pt_at(g::RegularGrid{<: GO.Spherical}, i::Int, j::Int) =
    GO.UnitSphereFromGeographic()((g.x[i], g.y[j]))
@inline _pt_at(g::CellBasedGrid{<: GO.Spherical}, i::Int, j::Int) =
    @inbounds g.points[i, j]

# Perimeter iterator: walks the four sides of the [imin..imax] × [jmin..jmax]
# index rectangle in order (west column, east column, north row, south row),
# skipping degenerate sides. Yields the `_pt_at(grid, i, j)` for each border
# point. State is `(side::Int, k::Int)`; `side > 4` signals exhaustion.
struct PerimeterPoints{G}
    grid::G
    imin::Int
    imax::Int
    jmin::Int
    jmax::Int
end

Base.IteratorSize(::Type{<: PerimeterPoints}) = Base.HasLength()
function Base.length(p::PerimeterPoints)
    nrow = p.imax - p.imin + 1
    ncol = p.jmax - p.jmin + 1
    n = ncol                                       # west column
    n += (p.imax != p.imin) * ncol                 # east column
    n += (p.jmax != p.jmin) * nrow * 2             # north + south rows
    return n
end
Base.eltype(::Type{PerimeterPoints{G}}) where {G} =
    Core.Compiler.return_type(_pt_at, Tuple{G, Int, Int})

@inline function Base.iterate(p::PerimeterPoints, state = (1, p.jmin))
    while true
        side, k = state
        if side == 1                               # west: (imin, k)
            k <= p.jmax && return _pt_at(p.grid, p.imin, k), (1, k + 1)
            state = (2, p.jmin)
        elseif side == 2                           # east: (imax, k)
            (p.imax != p.imin && k <= p.jmax) &&
                return _pt_at(p.grid, p.imax, k), (2, k + 1)
            state = (3, p.imin)
        elseif side == 3                           # north: (k, jmax)
            (p.jmax != p.jmin && k <= p.imax) &&
                return _pt_at(p.grid, k, p.jmax), (3, k + 1)
            state = (4, p.imin)
        elseif side == 4                           # south: (k, jmin)
            (p.jmax != p.jmin && k <= p.imax) &&
                return _pt_at(p.grid, k, p.jmin), (4, k + 1)
            return nothing
        else
            return nothing
        end
    end
end

"""
    circle_from_four_corners(corner_points, other_points)

Bounding `SphericalCap` covering 4 cell corners passed in (BL, TL, BR, TR) order
plus an iterable of additional perimeter points. `corner_points` and the
elements of `other_points` may be `UnitSphericalPoint`s or `(lon, lat)` tuples
— `UnitSphereFromGeographic` is a no-op on already-converted points.

Public for use by extensions (e.g. `HealpixExt`) that build trees out of
arbitrary 4-corner polygons. Internally a thin wrapper over [`_spherical_cap`].
"""
function circle_from_four_corners(corner_points, other_points)
    raw = GO.UnitSphereFromGeographic().(corner_points)
    # Reorder (BL, TL, BR, TR) → CCW (SW, SE, NE, NW) for slerp midpoints.
    p1, p2, p3, p4 = raw[1], raw[3], raw[4], raw[2]
    return _spherical_cap(p1, p2, p3, p4,
        (GO.UnitSphereFromGeographic()(p) for p in other_points))
end

# Build a SphericalCap covering 4 CCW corners (SW, SE, NE, NW), the slerp
# midpoints of their 4 edges (great-circle bulge), and every point yielded by
# `perimeter` (constant-lat/lon bulge along cell sides).
@inline function _spherical_cap(p1, p2, p3, p4, perimeter)
    center = LinearAlgebra.normalize((p1 + p2 + p3 + p4) / 4)
    p12 = GO.UnitSpherical.slerp(p1, p2, 0.5)
    p23 = GO.UnitSpherical.slerp(p2, p3, 0.5)
    p34 = GO.UnitSpherical.slerp(p3, p4, 0.5)
    p41 = GO.UnitSpherical.slerp(p4, p1, 0.5)
    d = max(
        GO.spherical_distance(center, p1), GO.spherical_distance(center, p2),
        GO.spherical_distance(center, p3), GO.spherical_distance(center, p4),
        GO.spherical_distance(center, p12), GO.spherical_distance(center, p23),
        GO.spherical_distance(center, p34), GO.spherical_distance(center, p41),
    )
    for p in perimeter
        d = max(d, GO.spherical_distance(center, p))
    end
    # The 1.0001 slack guards against missed intersections from rounding error.
    return GO.UnitSpherical.SphericalCap(center, d * 1.0001)
end

function cell_range_extent(q::CellBasedGrid{<: GO.Spherical}, irange::UnitRange{Int}, jrange::UnitRange{Int})
    imin, imax = extrema(irange); imax += 1
    jmin, jmax = extrema(jrange); jmax += 1
    return _spherical_cap(
        _pt_at(q, imin, jmin), _pt_at(q, imax, jmin),
        _pt_at(q, imax, jmax), _pt_at(q, imin, jmax),
        PerimeterPoints(q, imin, imax, jmin, jmax),
    )
end

function cell_range_extent(q::RegularGrid{<: GO.Spherical}, irange::UnitRange{Int}, jrange::UnitRange{Int})
    imin, imax = extrema(irange); imax += 1
    jmin, jmax = extrema(jrange); jmax += 1
    return _spherical_cap(
        _pt_at(q, imin, jmin), _pt_at(q, imax, jmin),
        _pt_at(q, imax, jmax), _pt_at(q, imin, jmax),
        PerimeterPoints(q, imin, imax, jmin, jmax),
    )
end