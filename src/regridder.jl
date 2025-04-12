const DEFAULT_NODECAPACITY = 10
const DEFAULT_FLOATTYPE = Float64
const DEFAULT_MANIFOLD = GeometryOps.Planar()
const DEFAULT_MATRIX = SparseArrays.spzeros # SparseCSC for regridder

abstract type AbstractRegridder end

struct Regridder{W, A} <: AbstractRegridder
    intersections :: W # Matrix of area intersections between cells on the source and destination grid
    dst_areas :: A     # Vector of areas on the destination grid
    src_areas :: A     # Vector of areas on the source grid
end

Base.size(regridder::Regridder, args...; kwargs...) = size(regridder.intersections, args...; kwargs...)

# allocate the areas matrix as SparseCSC if not provided
function intersection_areas(
    src_field, # arrays of polygons of the first grid
    dst_field, # arrays of polygons of the second grid
    m::GeometryOps.Manifold = DEFAULT_MANIFOLD;
    T::Type{<:Number} = DEFAULT_FLOATTYPE,          # float type used for the areas matrix = regridder
    kwargs...
)
    # unless `areas::AbstractMatrix` is provided (see in-place method ! below), create a SparseCSC matrix
    areas = DEFAULT_MATRIX(T, length(src_field), length(dst_field))
    return compute_intersection_areas!(areas, src_field, dst_field, m; kwargs...)
end

function compute_intersection_areas!(
    areas::AbstractMatrix,      # intersection areas for all combinatations of grid cells in grid1, grid2
    grid1,                      # arrays of polygons
    grid2,                      # arrays of polygons
    m::GeometryOps.Manifold = GeometryOps.Planar();
    nodecapacity1 = DEFAULT_NODECAPACITY,
    nodecapacity2 = DEFAULT_NODECAPACITY,
)
    # Prepare STRtrees for the two grids, to speed up intersection queries
    # we may want to separately tune nodecapacity if one is much larger than the other.  
    # specifically we may want to tune leaf node capacity via Hilbert packing while still 
    # constraining inner node capacity.  But that can come later.
    tree1 = SortTileRecursiveTree.STRtree(grid1; nodecapacity = nodecapacity1) 
    tree2 = SortTileRecursiveTree.STRtree(grid2; nodecapacity = nodecapacity2)
    # Do the dual query, which is the most efficient way to do this,
    # by iterating down both trees simultaneously, rejecting pairs of nodes that do not intersect.
    # when we find an intersection, we calculate the area of the intersection and add it to the result matrix.
    GeometryOps.SpatialTreeInterface.do_dual_query(Extents.intersects, tree1, tree2) do i1, i2
        p1, p2 = grid1[i1], grid2[i2]
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

        area_of_intersection = GeometryOps.area(m, intersection_polys)
        if area_of_intersection > 0
            areas[i1, i2] += area_of_intersection
        end
    end

    return areas
end

"""$(TYPEDSIGNATURES)
Return a Regridder that transfers data from `src_field` to `dst_field`.

Regridder stores the intersection areas between
The areas are computed by summing the regridder along the first and second dimensions
as the regridder is a matrix of the intersection areas between each grid cell between the
two grids."""
function Regridder(
    src_vertices,
    dst_vertices;
    FT = Float64,
    AT = SparseArrays.SparseMatrixCSC,
)
    # TODO: make this work
    # intersections = intersection_areas(AT{FT}, src_vertices, dst_vertices)
    intersections = intersection_areas(src_vertices, dst_vertices)
    src_areas = cell_areas(intersec_area, dims = 1)
    dst_areas = cell_areas(intersec_area, dims = 2)
    return Regridder(intersec_area, src_areas, dst_areas)
end
    
function compute_weights!(regridder::Regridder, src_vertices, dst_vertices)
    compute_intersection_areas!(regridder.intersections, src_vertices, dst_vertices)
    return regridder
end

"""$(TYPEDSIGNATURES)
Returns area vectors (out, in) for the grids used to create the regridder.
The area vectors are computed by summing the regridder along the first and second dimensions
as the regridder is a matrix of the intersection areas between each grid cell between the
two grids."""
cell_area(weights::AbstractMatrix) = cell_area(weights, :out), cell_area(weights, :in)

"""$(TYPEDSIGNATURES) Area vector from `regridder`, `dims` can be `:in` or `:out`."""
Base.@propagate_inbounds cell_areas(intersections::AbstractMatrix; dims) = vec(sum(intersections; dims))
    