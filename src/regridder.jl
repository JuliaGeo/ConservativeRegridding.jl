using SortTileRecursiveTree

const DEFAULT_NODECAPACITY = 10
const DEFAULT_MANIFOLD = GeometryOps.Planar()
const DEFAULT_FLOATTYPE = Float64
const DEFAULT_MATRIX = SparseArrays.SparseMatrixCSC{DEFAULT_FLOATTYPE}
const DEFAULT_MATRIX_CONSTRUCTOR = SparseArrays.spzeros # SparseCSC for regridder

abstract type AbstractRegridder end

struct Regridder{W, A} <: AbstractRegridder
    intersections :: W # Matrix of area intersections between cells on the source and destination grid
    dst_areas :: A     # Vector of areas on the destination grid
    src_areas :: A     # Vector of areas on the source grid
end

function Base.show(io::IO, regridder::Regridder{W, A}) where {W, A}
    n2, n1 = size(regridder)
    println(io, "$n2×$n1 Regridder{$W, $A}")
    Base.print_array(io, regridder.intersections)
    println(io, "\n\nSource areas: ", regridder.src_areas)
    print(io, "Dest.  areas: ", regridder.dst_areas)
end

"""$(TYPEDSIGNATURES)
Return a Regridder for the backwards regridding, i.e. from destination to source grid.
Does not copy any data, i.e. regridder for forward and backward share the same underlying arrays."""
LinearAlgebra.transpose(regridder::Regridder) =
    Regridder(transpose(regridder.intersections), regridder.src_areas, regridder.dst_areas)

Base.size(regridder::Regridder, args...) = size(regridder.intersections, args...)

function LinearAlgebra.normalize!(regridder::Regridder)
    (; intersections) = regridder
    norm = maximum(intersections)   # TODO is this the best normalizer?
    intersections ./= norm

    regridder.src_areas ./= norm
    regridder.dst_areas ./= norm
    return regridder
end

# allocate the areas matrix as SparseCSC if not provided
function intersection_areas(
    dst_field, # array of polygons of the target grid
    src_field; # array of polygons of the source grid
    T::Type{<:Number} = DEFAULT_FLOATTYPE,              # float type used for the areas matrix = regridder
    matrix_constructor = DEFAULT_MATRIX_CONSTRUCTOR,    # type of the areas matrix = regridder
    kwargs...
)
    # unless `areas::AbstractMatrix` is provided (see in-place method ! below), create a SparseCSC matrix
    areas = matrix_constructor(T, length(dst_field), length(src_field))
    return compute_intersection_areas!(areas, dst_field, src_field; kwargs...)
end

function compute_intersection_areas!(
    areas::AbstractMatrix,  # intersection areas for all combinatations of grid cells in field1, field2
    dst_grid,               # array of polygons
    src_grid;               # array of polygons
    manifold::GeometryOps.Manifold = DEFAULT_MANIFOLD,      # TODO currently not used
    dst_nodecapacity = DEFAULT_NODECAPACITY,
    src_nodecapacity = DEFAULT_NODECAPACITY,
)
    # Prepare STRtrees for the two grids, to speed up intersection queries
    # we may want to separately tune nodecapacity if one is much larger than the other.
    # specifically we may want to tune leaf node capacity via Hilbert packing while still
    # constraining inner node capacity.  But that can come later.
    tree1 = SortTileRecursiveTree.STRtree(src_grid; nodecapacity = src_nodecapacity)
    tree2 = SortTileRecursiveTree.STRtree(dst_grid; nodecapacity = dst_nodecapacity)
    # Do the dual query, which is the most efficient way to do this,
    # by iterating down both trees simultaneously, rejecting pairs of nodes that do not intersect.
    # when we find an intersection, we calculate the area of the intersection and add it to the result matrix.
    GeometryOps.SpatialTreeInterface.dual_depth_first_search(Extents.intersects, tree1, tree2) do i1, i2
        p1, p2 = src_grid[i1], dst_grid[i2]
        # may want to check if the polygons intersect first,
        # to avoid antimeridian-crossing multipolygons viewing a scanline.
        intersection_polys = try # can remove this now, got all the errors cleared up in the fix.
            # At some future point, we may want to add the manifold here
            # but for right now, GeometryOps only supports planar polygons anyway.
            GeometryOps.intersection(p1, p2; target = GeoInterface.PolygonTrait())
        catch e
            @error "Intersection failed!" i1 i2
            rethrow(e)
        end

        area_of_intersection = GeometryOps.area(intersection_polys)
        if area_of_intersection > 0
            areas[i2, i1] += area_of_intersection
        end
    end

    return areas
end

get_vertices(polygons) = polygons       # normally vector of polygons, which are vectors of points
get_vertices(polygons::AbstractMatrix) = eachcol(polygons)

"""$(TYPEDSIGNATURES)
Return a Regridder that transfers data from `src_field` to `dst_field`.

Regridder stores the intersection areas between
The areas are computed by summing the regridder along the first and second dimensions
as the regridder is a matrix of the intersection areas between each grid cell between the
two grids. Additional `kwargs` are passed to the `intersection_areas` function.
"""
function Regridder(
    dst_vertices, # assumes something like this ::AbstractVector{<:AbstractVector{Tuple}},
    src_vertices;
    normalize::Bool = true,
    kwargs...
)
    # wrap into GeoInterface.Polygon, apply antimeridian cuttng via fix
    dst_polys = GeoInterface.Polygon.(GeoInterface.LinearRing.(get_vertices(dst_vertices))) .|> GeometryOps.fix
    src_polys = GeoInterface.Polygon.(GeoInterface.LinearRing.(get_vertices(src_vertices))) .|> GeometryOps.fix

    intersections = intersection_areas(dst_polys, src_polys; kwargs...)

    # If the two grids completely overlap, then the areas should be equivalent
    # to the sum of the intersection areas along the second and fisrt dimensions,
    # for src and dst, respectively. This is not the case if the two grids do not cover the same area.
    dst_areas = GeometryOps.area.(dst_polys)
    src_areas = GeometryOps.area.(src_polys)

    regridder = Regridder(intersections, dst_areas, src_areas)
    normalize && LinearAlgebra.normalize!(regridder)
    return regridder
end
