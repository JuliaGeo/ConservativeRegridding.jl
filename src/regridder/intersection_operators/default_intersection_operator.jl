import GeometryOps as GO
import GeoInterface as GI
using GridInterface: cellshape, CellShape, ConvexCells, GeneralCells

"""
Error type to capture the polygons that failed to intersect, as well as the initial error thrown.
"""
struct DefaultIntersectionFailureError <: Base.Exception
    p1
    p2
    e
end

function Base.showerror(io::IO, e::DefaultIntersectionFailureError)
    print(io, "Intersection failed with the following error.  Capture this error object and access `err.p1` and `err.p2` to access the polygons that failed to intersect.")
    Base.showerror(io, e.e)
end

"The clipper a [`DefaultIntersectionOperator`](@ref) runs: convex-convex Sutherland-Hodgman or the general Foster-Hormann clipper."
abstract type ClipKind end
"Both grids have cells of a few straight edges (`ConvexCells` or `SimpleCells`): convex-convex Sutherland-Hodgman on the sphere."
struct ConvexClip <: ClipKind end
"At least one grid has `GeneralCells`: GeometryOps' general clipper."
struct GeneralClip <: ClipKind end

"""
    DefaultIntersectionOperator(manifold, src, dst)
    DefaultIntersectionOperator(manifold)

Intersection operator that computes the *area* of intersection between a source and a
destination cell.  This is the operator that [`Regridder`](@ref) uses by default, and it
produces the standard first-order conservative regridding weights.

The clipper follows the manifold and the `cellshape` of the two grids: on `Planar`
Foster-Hormann clipping; on `Spherical` convex-convex Sutherland-Hodgman when both
grids declare `ConvexCells` or `SimpleCells` (`DefaultIntersectionOperator(manifold)`
assumes so; the quadrilateral and HEALPix-like cells of the shipped grids, which the
convex clip has always measured), and GeometryOps' general Foster-Hormann clipper on
the sphere when either grid has `GeneralCells`. (The general spherical clipper is not
used for `SimpleCells`: on coincident cell edges, lat-lon against HEALPix say, it
returns wrong areas, where the convex clip is exact.) It uses all the defaults of the
intersection-operator interface ([`OutOfPlaceSingleResult`](@ref) return style,
`Float64` element type).

On `Spherical`, [`task_local_operator`](@ref) hands each assembly task a copy carrying
a private clipping-buffer cache, reused by repeated builds on that task. Caches must not
be shared across tasks.
"""
struct DefaultIntersectionOperator{M, K <: ClipKind, C}
    manifold::M
    kind::K
    cache::C
end
DefaultIntersectionOperator(manifold) = DefaultIntersectionOperator(manifold, ConvexClip(), nothing)
DefaultIntersectionOperator(manifold, src, dst) =
    DefaultIntersectionOperator(manifold, clip_kind(cellshape(src), cellshape(dst)), nothing)

"The [`ClipKind`](@ref) for a pair of cell shapes: the general clipper only when either side has `GeneralCells`."
clip_kind(::GeneralCells, ::CellShape) = GeneralClip()
clip_kind(::CellShape, ::GeneralCells) = GeneralClip()
clip_kind(::GeneralCells, ::GeneralCells) = GeneralClip()
clip_kind(::CellShape, ::CellShape) = ConvexClip()

function (op::DefaultIntersectionOperator{<: GeometryOps.Planar})(p1, p2)
    intersection_polys = #=try; =#
        GeometryOps.intersection(GO.FosterHormannClipping(GO.Planar()), p1, p2; target = GeoInterface.PolygonTrait())
    # catch
    #     throw(DefaultIntersectionFailureError(p1, p2, e))
    # end
    return GeometryOps.area(GO.Planar(), intersection_polys)
end

function (op::DefaultIntersectionOperator{M, ConvexClip})(p1, p2) where {M <: GeometryOps.Spherical}
    alg = GeometryOps.ConvexConvexSutherlandHodgman(op.manifold)
    result = #=try; =#
        _sutherland_hodgman_intersection_area(alg, p1, p2, op.cache)
    # catch
    #     throw(DefaultIntersectionFailureError(p1, p2, e))
    # end
    return result
end

# The general spherical clipper: Foster-Hormann on the sphere, then the area of
# the result on this manifold (its radius).
function (op::DefaultIntersectionOperator{M, GeneralClip})(p1, p2) where {M <: GeometryOps.Spherical}
    intersection_polys = GeometryOps.intersection(
        GO.FosterHormannClipping(op.manifold), p1, p2; target = GeoInterface.PolygonTrait())
    return GeometryOps.area(op.manifold, intersection_polys)
end

_sutherland_hodgman_intersection_area(alg, p1, p2, ::Nothing) =
    GeometryOps.intersection_area(alg, p1, p2)
_sutherland_hodgman_intersection_area(alg, p1, p2, cache) =
    GeometryOps.intersection_area(alg, p1, p2; cache)

# `task_local_operator` may be reached through a wrapping custom operator, so the clip
# cache is kept directly on the current task rather than in assembly-specific plumbing.
# Scheduler migration is safe; distinct tasks never share a mutable cache.
struct _SphericalClipCacheKey{M} end

function _task_local_clip_cache(manifold::M) where {M}
    tls = task_local_storage()
    cache = get(tls, _SphericalClipCacheKey{M}, nothing)
    if cache === nothing
        cache = GeometryOps.SutherlandHodgmanCache(manifold)
        tls[_SphericalClipCacheKey{M}] = cache
    end
    return cache
end

function task_local_operator(op::DefaultIntersectionOperator{<: GeometryOps.Spherical, ConvexClip})
    cache = _task_local_clip_cache(op.manifold)
    return DefaultIntersectionOperator(op.manifold, op.kind, cache)
end
