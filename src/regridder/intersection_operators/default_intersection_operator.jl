import GeometryOps as GO
import GeoInterface as GI

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

"""
    DefaultIntersectionOperator(manifold::GeometryOps.Manifold)

Intersection operator that computes the *area* of intersection between a source and a
destination cell.  This is the operator that [`Regridder`](@ref) uses by default, and it
produces the standard first-order conservative regridding weights.

Dispatches to the appropriate intersection algorithm based on the manifold:
Foster-Hormann clipping on `Planar`, and convex-convex Sutherland-Hodgman on `Spherical`.
It uses all the defaults of the intersection-operator interface
([`OutOfPlaceSingleResult`](@ref) return style, `Float64` element type).

On `Spherical`, [`task_local_operator`](@ref) hands each assembly task a copy carrying
a private clipping-buffer cache; caches must not be shared across tasks.
"""
struct DefaultIntersectionOperator{M, C}
    manifold::M
    cache::C
end
DefaultIntersectionOperator(manifold) = DefaultIntersectionOperator(manifold, nothing)

function (op::DefaultIntersectionOperator{<: GeometryOps.Planar})(p1, p2)
    intersection_polys = #=try; =#
        GeometryOps.intersection(GO.FosterHormannClipping(GO.Planar()), p1, p2; target = GeoInterface.PolygonTrait())
    # catch
    #     throw(DefaultIntersectionFailureError(p1, p2, e))
    # end
    return GeometryOps.area(GO.Planar(), intersection_polys)
end

function (op::DefaultIntersectionOperator{M})(p1, p2) where {M <: GeometryOps.Spherical}
    alg = GeometryOps.ConvexConvexSutherlandHodgman(op.manifold)
    intersection_polys = #=try; =#
        _sutherland_hodgman_intersection(alg, p1, p2, op.cache)
    # catch
    #     throw(DefaultIntersectionFailureError(p1, p2, e))
    # end
    return GeometryOps.area(op.manifold, intersection_polys)
end

_sutherland_hodgman_intersection(alg, p1, p2, ::Nothing) =
    GeometryOps.intersection(alg, p1, p2; target = GeoInterface.PolygonTrait())
_sutherland_hodgman_intersection(alg, p1, p2, cache) =
    GeometryOps.intersection(alg, p1, p2; target = GeoInterface.PolygonTrait(), cache)

function task_local_operator(op::DefaultIntersectionOperator{<: GeometryOps.Spherical})
    cache = GeometryOps.SutherlandHodgmanCache(op.manifold)
    return DefaultIntersectionOperator(op.manifold, cache)
end
