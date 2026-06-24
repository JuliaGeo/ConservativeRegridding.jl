#=
# Intersection-grid assembler

In this example, we'll use the regridder construction machinery to build a sparse matrix
of the raw polygons of intersection between two grids, so `mat[i, j]` holds the actual polygon.

## Why?

The idea behind this is to be able to assemble a sparse matrix of the raw polygons of intersection
between two grids.  In climate modeling, this can be particularly useful for the regridding of fluxes,
and also preserves the geometric structure of the two previous grids (via the rows and columns).

From here, it's possible to build the first-order conservative regridder without a second tree search and
intersection computation, which may speed things up significantly.

## How?

The way that we do this is by defining a custom intersection operator, and in fact that is all we have to do.
In this case, for convenience, we'll use the `OutOfPlaceSingleResult` return style.
=#

# ## Implementation

import ConservativeRegridding: IntersectionReturnStyle, OutOfPlaceSingleResult, 
                               output_eltype, should_store_result
import GeometryOpsCore: Manifold, Planar, Spherical

import GeoInterface as GI, GeometryOps as GO

# This is a simple helper function to get a polygon type out.
_get_poly_type(point_type = Tuple{T, T}, z = false) =
    GI.Polygon{z, false, Vector{GI.LinearRing{z, false, Vector{point_type}, Nothing, Nothing}}, Nothing, Nothing}

# Now let's define the intersection operator, and implement the ConservativeRegridding intersection operator
# interface.

struct IntersectionGridOperator{M <: Manifold, PolygonType}
    manifold::M
end
function IntersectionGridOperator(manifold::Planar)
    polygon_type = _get_poly_type(Tuple{Float64, Float64})
    return IntersectionGridOperator{Planar, polygon_type}(manifold)
end
function IntersectionGridOperator(manifold::S) where {S <: Spherical}
    polygon_type = _get_poly_type(GO.UnitSpherical.UnitSphericalPoint{Float64}, true)
    return IntersectionGridOperator{S, polygon_type}(manifold)
end

IntersectionReturnStyle(::IntersectionGridOperator) = OutOfPlaceSingleResult()

## NB: this assumes the function returns `nothing` when there is no intersection.
## That _should_ be inferrable by the Julia compiler, thus avoiding dynamic dispatch.
should_store_result(op::IntersectionGridOperator, result::Nothing) = false
should_store_result(op::IntersectionGridOperator, result::GI.Polygon) = true

output_eltype(::IntersectionGridOperator{M, P}) where {M, P} = P

# Now that we've defined the interface, the only thing left to do is to implement the actual intersection logic.

function (op::IntersectionGridOperator{<: Planar, P})(src_cell, dst_cell) where {P}
    # NB: this assumes there can only ever be either one or zero intersection polygons
    intersection_polys = GO.intersection(GO.FosterHormannClipping(GO.Planar()), src_cell, dst_cell; target = GI.PolygonTrait())
    isempty(intersection_polys) && return nothing
    return only(intersection_polys)
end

function (op::IntersectionGridOperator{<: Spherical, P})(src_cell, dst_cell) where {P}
    intersection_poly = GO.intersection(GO.ConvexConvexSutherlandHodgman(op.manifold), src_cell, dst_cell; target = GI.PolygonTrait())
    if iszero(GO.area(op.manifold, intersection_poly))
        return nothing
    else
        return intersection_poly
    end
end

# ## Try it out!

using Oceananigans: LatitudeLongitudeGrid, RotatedLatitudeLongitudeGrid
using ConservativeRegridding: intersection_areas, Regridder


lonlat_grid = LatitudeLongitudeGrid(size=(36, 18, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))
rotated_grid = RotatedLatitudeLongitudeGrid(size=(36, 18, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1), north_pole=(70, 55))

regridder = Regridder(
    GO.Spherical(), lonlat_grid, rotated_grid; 
    intersection_operator = IntersectionGridOperator(GO.Spherical()),
    normalize = false,
    threaded = false,
)

# Note that this is now a matrix of polygons!

using SparseArrays
using GeoMakie, GLMakie

unitspherical_polygons = SparseArrays.nonzeros(regridder.intersections)
latlong_polygons = GO.transform(GO.UnitSpherical.GeographicFromUnitSphere(), unitspherical_polygons)

f, a, p = poly(latlong_polygons; strokewidth = 1, color = rand(RGBf, length(latlong_polygons)), axis = (; type = GlobeAxis))
meshimage!(a, -180..180, -90..90, reshape([colorant"white"], 1, 1); zlevel = -300_000) # background image
f