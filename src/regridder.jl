const DEFAULT_NODECAPACITY = 10
const DEFAULT_FLOATTYPE = Float64
const DEFAULT_MANIFOLD = GeometryOps.Planar()
const DEFAULT_MATRIX = SparseArrays.spzeros         # SparseCSC for regridder

# allocate the areas matrix as SparseCSC if not provided
function intersection_areas(
    grid1,                                          # arrays of polygons of the first grid
    grid2,                                          # arrays of polygons of the second grid
    m::GeometryOps.Manifold = DEFAULT_MANIFOLD;
    T::Type{<:Number} = DEFAULT_FLOATTYPE,          # float type used for the areas matrix = regridder
    kwargs...
)
    # unless `areas::AbstractMatrix` is provided (see in-place method ! below), create a SparseCSC matrix
    areas = DEFAULT_MATRIX(T, length(grid1), length(grid2))
    intersection_areas!(areas, grid1, grid2, m; kwargs...)
end

function intersection_areas!(
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

regridder(grid1, grid2) = intersection_areas(grid1, grid2)
regridder!(regridder::AbstractMatrix, grid1, grid2) = intersection_areas!(regridder, grid1, grid2)

"""$(TYPEDSIGNATURES)
Returns area vectors (out, in) for the grids used to create the regridder.
The area vectors are computed by summing the regridder along the first and second dimensions
as the regridder is a matrix of the intersection areas between each grid cell between the
two grids."""
cell_area(regridder::AbstractMatrix) = cell_area(regridder, :out), cell_area(regridder, :in)

"""$(TYPEDSIGNATURES) Area vector from `regridder`, `dims` can be `:in` or `:out`."""
Base.@propagate_inbounds function cell_area(regridder::AbstractMatrix, dims::Symbol)
    @boundscheck if !(dims in (:in, :out))
        throw(ArgumentError("Only accepts dims `:in` or `:out`; got $dims"))
    end
    
    # "in" is a sum along the 2nd dimension of the matrix, returning a vector of length of the 1st dimension (the output grid)
    if dims == :in
        vec(sum(regridder, dims = 1))
    elseif dims == :out
        vec(sum(regridder, dims = 2))
    end
end