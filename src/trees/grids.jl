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
struct ExplicitPolygonGrid{M <: GOCore.Manifold, PolyMatrixType <: AbstractMatrix} <: AbstractCurvilinearGrid{M}
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
struct CellBasedGrid{M <: GOCore.Manifold, PointMatrixType <: AbstractMatrix} <: AbstractCurvilinearGrid{M}
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
struct RegularGrid{M <: GOCore.Manifold, DX, DY} <: AbstractCurvilinearGrid{M}
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
    # Max great-circle angle = acos(min cosine): dot products + one acos, no per-point trig.
    radius = acos(clamp(minimum(p -> LinearAlgebra.dot(center, p), all_points), -1.0, 1.0))
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

The bounding cap must cover the four corners of the index rectangle and the points
sampled along its perimeter — cell edges follow constant-lat / constant-lon, not
great circles, so they bulge differently. Earlier versions materialized the perimeter
into a fresh `Vector` per call, which drove most of the GC pressure in the dual-DFS
hot path.

Instead each vertex is accessed on demand through the per-grid `getvertex` interface
method, and the border is walked with the stack-resident `CurvilinearGridPerimeterPoints`
iterator. The cap builder folds `max`-distance over the corners and that iterator — no
buffers, no shared state, threadsafe by construction.
=#

# `getvertex` interface methods (interfaces.jl) for the structured spherical grids —
# the one piece of per-grid variation the bounding-cap code below dispatches on.
@inline getvertex(g::RegularGrid{<: GO.Spherical}, i::Int, j::Int) =
    GO.UnitSphereFromGeographic()((g.x[i], g.y[j]))
@inline getvertex(g::CellBasedGrid{<: GO.Spherical}, i::Int, j::Int) =
    @inbounds g.points[i, j]

# Lazy iterator over the border-ring vertices of the point block
# [imin..imax] × [jmin..jmax], in order: west column, east column, then the
# interiors of the south and north rows (so each of the 4 corners is yielded once,
# by the columns). Built on `getvertex`, so it works for any grid implementing the
# interface. State is a single linear index `n`; the `n → (i,j)` map is plain
# arithmetic and iteration ends once `n` passes `length`.
struct CurvilinearGridPerimeterPoints{G}
    grid::G
    imin::Int
    imax::Int
    jmin::Int
    jmax::Int
end

Base.IteratorSize(::Type{<: CurvilinearGridPerimeterPoints}) = Base.HasLength()
# A W×H point block has 2W + 2H − 4 border vertices (corners counted once).
Base.length(p::CurvilinearGridPerimeterPoints) =
    2 * (p.imax - p.imin + 1) + 2 * (p.jmax - p.jmin + 1) - 4

@inline function Base.iterate(p::CurvilinearGridPerimeterPoints, n::Int = 1)
    n > length(p) && return nothing
    H = p.jmax - p.jmin + 1
    W = p.imax - p.imin + 1
    if n <= H
        i, j = p.imin, p.jmin + n - 1                # west column (incl. both W corners)
    elseif n <= 2H
        i, j = p.imax, p.jmin + (n - H) - 1          # east column (incl. both E corners)
    elseif n <= 2H + (W - 2)
        i, j = p.imin + (n - 2H), p.jmin             # south row interior
    else
        i, j = p.imin + (n - 2H - (W - 2)), p.jmax   # north row interior
    end
    return getvertex(p.grid, i, j), n + 1
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

# cos∠(center, great-circle midpoint of a, b). slerp(a, b, 0.5) == normalize(a + b), so the
# cosine is dot(center, a+b)/‖a+b‖ — no trig. Returns 1 (i.e. no contribution to the min) when
# a, b are ~antipodal and the midpoint is ill-defined; this happens on a large index-rectangle's
# edge spanning ~180°, where the two endpoints already force the cap wide, so it's redundant.
@inline function _midcos(center, a, b)
    s = a + b
    n2 = LinearAlgebra.dot(s, s)
    return n2 < 1e-12 ? one(n2) : LinearAlgebra.dot(center, s) / sqrt(n2)
end

# Build a SphericalCap covering 4 CCW corners (SW, SE, NE, NW), their 4 great-circle edge
# midpoints (edge bulge), and every point yielded by `perimeter` (constant-lat/lon bulge along
# cell sides). Trig-free: the max great-circle angle is acos of the min cosine to those points.
@inline function _spherical_cap(p1, p2, p3, p4, perimeter)
    center = LinearAlgebra.normalize((p1 + p2 + p3 + p4) / 4)
    cosmin = min(
        LinearAlgebra.dot(center, p1), LinearAlgebra.dot(center, p2),
        LinearAlgebra.dot(center, p3), LinearAlgebra.dot(center, p4),
        _midcos(center, p1, p2), _midcos(center, p2, p3),
        _midcos(center, p3, p4), _midcos(center, p4, p1),
    )
    for p in perimeter
        cosmin = min(cosmin, LinearAlgebra.dot(center, p))
    end
    # The 1.0001 slack guards against missed intersections from rounding error.
    return GO.UnitSpherical.SphericalCap(center, acos(clamp(cosmin, -1.0, 1.0)) * 1.0001)
end

# Bounding cap for a range of cells on any spherical curvilinear grid: the four
# corners, their great-circle edge midpoints, and the perimeter vertices of the
# index rectangle. One generic method — concrete grids supply only `getvertex`.
# (`ExplicitPolygonGrid{<: GO.Spherical}` keeps its own, more-specific method above.)
function cell_range_extent(g::AbstractCurvilinearGrid{<: GO.Spherical}, irange::UnitRange{Int}, jrange::UnitRange{Int})
    imin, imax = extrema(irange); imax += 1
    jmin, jmax = extrema(jrange); jmax += 1
    return _spherical_cap(
        getvertex(g, imin, jmin), getvertex(g, imax, jmin),
        getvertex(g, imax, jmax), getvertex(g, imin, jmax),
        CurvilinearGridPerimeterPoints(g, imin, imax, jmin, jmax),
    )
end
