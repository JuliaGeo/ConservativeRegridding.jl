import GeoInterface as GI
import GeometryOpsCore as GOCore
using GridInterface: UnitSphericalPoint

"""
    asgrid(x) -> GridInterface grid
    asgrid(manifold, x)

The grid [`Regridder`](@ref) regrids for an input `x`: a GridInterface grid is
its own; a bare corner matrix, axis pair or polygon collection becomes one of
CR's own grids ([`CellBasedGrid`](@ref GridInterface.Structured.CellBasedGrid),
[`RegularGrid`](@ref GridInterface.Structured.RegularGrid),
[`ExplicitPolygonGrid`](@ref GridInterface.Structured.ExplicitPolygonGrid),
[`PolygonVectorGrid`](@ref)); the package extensions define it for their
native objects (an Oceananigans grid or field, a ClimaCore cubed-sphere
topology, space or field, a `Healpix.HealpixMap`, a RingGrids grid or field),
returning their GridInterface providers.

The manifold form builds CR's own grids on `manifold` (the two-argument
`Regridder` guesses it from the input: `UnitSphericalPoint` corners or
vertices mean `Spherical()`, anything else `Planar()`); for every other
input it is `asgrid(x)`.

- `AbstractMatrix` of points: the `(n+1) × (m+1)` corners of an `n × m` lattice,
  `CellBasedGrid(manifold, points)` (periodic and collapsed seams detected by
  tolerance; build the grid yourself to declare them);
- `AbstractMatrix` of polygons: `ExplicitPolygonGrid(manifold, polygons)`;
- `(x, y)` axis vectors: `RegularGrid(manifold, x, y)`;
- `AbstractVector` of polygons: `PolygonVectorGrid(manifold, polygons)`.
"""
asgrid(x) = x
asgrid(::Manifold, x) = asgrid(x)
# A bare matrix or vector is only CR's when it holds polygons (or corner points):
# a native array type an extension answers for (a `HealpixMap` is a vector) goes
# to that extension's one-argument method, anything else is an error.
asgrid(x::AbstractMatrix) = _isbare(x) ? asgrid(_guess_manifold(x), x) : _notagrid(x)
asgrid(m::Manifold, cells::AbstractMatrix) =
    _ispolygons(cells) ? ExplicitPolygonGrid(m, cells) :
    _ispoints(cells) ? CellBasedGrid(m, cells) : asgrid(cells)
asgrid(axes::NTuple{2,AbstractVector{<:Real}}) = asgrid(Planar(), axes)
asgrid(m::Manifold, axes::NTuple{2,AbstractVector{<:Real}}) = RegularGrid(m, axes[1], axes[2])
asgrid(x::AbstractVector) = _ispolygons(x) ? asgrid(_guess_manifold(x), x) : _notagrid(x)
asgrid(m::Manifold, polygons::AbstractVector) =
    _ispolygons(polygons) ? PolygonVectorGrid(m, polygons) : asgrid(polygons)

_notagrid(x) = throw(ArgumentError(
    "cannot regrid a $(typeof(x)): pass a GridInterface grid, a matrix of corner points or " *
    "polygons, a vector of polygons, an `(x, y)` axis pair, or a native object a loaded " *
    "ConservativeRegridding extension answers for"))
_ispolygons(x) = !isempty(x) && GI.trait(first(x)) isa Union{GI.PolygonTrait,GI.MultiPolygonTrait}
_ispoints(x) = !isempty(x) && GI.trait(first(x)) isa GI.PointTrait
_isbare(x) = _ispolygons(x) || _ispoints(x)
_isunitspherical(p) = p isa UnitSphericalPoint
_isunitspherical(::Type{<:UnitSphericalPoint}) = true
_isunitspherical(::Type) = false
function _guess_manifold(x)
    _isunitspherical(eltype(x)) && return Spherical()
    isempty(x) && return Planar()
    first_cell = first(x)
    if GI.trait(first_cell) isa GI.MultiPolygonTrait
        first_cell = GI.getgeom(first_cell, 1)
    end
    if GI.trait(first_cell) isa GI.PolygonTrait
        return _isunitspherical(GI.getpoint(GI.getexterior(first_cell), 1)) ? Spherical() : Planar()
    end
    return _isunitspherical(first_cell) ? Spherical() : Planar()
end

# The manifold the two-argument `Regridder` computes on for a bare input: CR's
# guess for corner matrices, axis pairs and polygon collections, `best_manifold`
# (the grid's own, or an extension's answer for a native object) otherwise.
input_manifold(x) = best_manifold(x)
input_manifold(x::AbstractMatrix) = _isbare(x) ? _guess_manifold(x) : best_manifold(x)
input_manifold(x::AbstractVector) = _ispolygons(x) ? _guess_manifold(x) : best_manifold(x)
input_manifold(::NTuple{2,AbstractVector{<:Real}}) = Planar()

"""
    PolygonVectorGrid(manifold, polygons::AbstractVector)

A vector of polygons as a GridInterface grid: cell `i` is `polygons[i]`
(`GeneralCells`, so every verb comes from the geometry), on `manifold`
(`UnitSphericalPoint` vertices on `Spherical()`). Its tree is GridInterface's
generic STR R-tree over the cell extents, built on each `treeify`; wrap the
grid in [`GridAndTree`](@ref GridInterface.GridAndTree) to reuse one tree, or
pair it with any SpatialTreeInterface tree over the same indices (an
`STRtree` of the polygons).
"""
struct PolygonVectorGrid{M<:Manifold,V<:AbstractVector} <: GridInterface.AbstractGrid
    manifold::M
    polygons::V
end
PolygonVectorGrid(polygons::AbstractVector) = PolygonVectorGrid(_guess_manifold(polygons), polygons)
GridInterface.ncells(g::PolygonVectorGrid) = length(g.polygons)
GOCore.best_manifold(g::PolygonVectorGrid) = g.manifold
GridInterface.cellshape(::PolygonVectorGrid) = GridInterface.GeneralCells()
GridInterface.cell_geometry(g::PolygonVectorGrid, i::Integer) = g.polygons[i]
GridInterface.cell_geometry_is_expensive(::PolygonVectorGrid) = false
Base.show(io::IO, g::PolygonVectorGrid) =
    print(io, "PolygonVectorGrid(", length(g.polygons), " polygons on ", g.manifold, ")")
