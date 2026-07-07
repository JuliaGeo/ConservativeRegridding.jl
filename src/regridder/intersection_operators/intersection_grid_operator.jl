import GeometryOps as GO
import GeoInterface as GI

"""
    IntersectionGridOperator(manifold::GeometryOps.Manifold)

Intersection operator that stores the raw *polygons* of intersection between source and
destination cells, rather than their areas.  

Pass it to [`Regridder`](@ref) via the `intersection_operator` keyword (with `normalize = false`), 
and `regridder.intersections` will then be a sparse matrix whose entry `[i, j]` holds the polygon of 
intersection between destination cell `i` and source cell `j`.

Since this preserves the exact geometry of intersection between the source and target grids, 
it's notably useful for determining flux transport during regridding in climate models.

!!! warning
    In the spherical domain, this operator only supports convex cells.
"""
struct IntersectionGridOperator{M <: Manifold, PolygonType}
    manifold::M
end

IntersectionGridOperator(manifold::M) where {M <: Manifold} =
    IntersectionGridOperator{M, intersection_polygon_type(manifold)}(manifold)

"""
    intersection_polygon_type(manifold) -> Type

Concrete `GeoInterface.Polygon` type produced by clipping two cells on `manifold`.
Stored as the `PolygonType` parameter of [`IntersectionGridOperator`](@ref), so that
[`output_eltype`](@ref) yields a concretely typed sparse matrix.
"""
intersection_polygon_type(::Planar) = _polygon_type(Tuple{Float64, Float64}, false)
intersection_polygon_type(::Spherical) = _polygon_type(GO.UnitSpherical.UnitSphericalPoint{Float64}, true)

_polygon_type(::Type{PointType}, z::Bool) where {PointType} =
    GI.Polygon{z, false, Vector{GI.LinearRing{z, false, Vector{PointType}, Nothing, Nothing}}, Nothing, Nothing}

IntersectionReturnStyle(::IntersectionGridOperator) = OutOfPlaceSingleResult()

output_eltype(::IntersectionGridOperator{M, PolygonType}) where {M, PolygonType} = PolygonType

# `nothing` marks "no intersection".  That is inferrable by the Julia compiler
# (`Union{Nothing, PolygonType}`), thus avoiding dynamic dispatch.
should_store_result(op::IntersectionGridOperator, result::Nothing) = false
should_store_result(op::IntersectionGridOperator, result::GI.Polygon) = true

function (op::IntersectionGridOperator{<: Planar})(src_cell, dst_cell)
    # NB: this assumes there can only ever be either one or zero intersection polygons,
    # which holds for convex cells.
    intersection_polys = GO.intersection(GO.FosterHormannClipping(GO.Planar()), src_cell, dst_cell; target = GI.PolygonTrait())
    isempty(intersection_polys) && return nothing
    return only(intersection_polys)
end

function (op::IntersectionGridOperator{<: Spherical})(src_cell, dst_cell)
    intersection_poly = GO.intersection(GO.ConvexConvexSutherlandHodgman(op.manifold), src_cell, dst_cell; target = GI.PolygonTrait())
    if iszero(GO.area(op.manifold, intersection_poly))
        return nothing
    else
        return intersection_poly
    end
end
