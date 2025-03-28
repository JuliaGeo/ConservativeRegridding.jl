
regridder(grid1, grid2; kwargs...) = area_of_intersection_operator(grid1, grid2; kwargs...)

regridder!(regridder::AbstractMatrix, grid1, grid2; kwargs...) = area_of_intersection_operator!(regridder, grid1, grid2; kwargs...)

"""$(TYPEDSIGNATURES)
Returns area vectors (out, in) for the grids used to create the regridder.
The area vectors are computed by summing the regridder along the first and second dimensions
as the regridder is a matrix of the intersection areas between each grid cell between the
two grids."""
areas(regridder::AbstractMatrix) = area(regridder, dims=:out), area(regridder, dims=:in)

"""$(TYPEDSIGNATURES) Area vector from `regridder`, `dims` can be `:in` or `:out`."""
area(regridder::AbstractMatrix; dims) = area(regridder, Val(dims))                  # pass on keyword as positional argument for dispatch

# "in" is a sum along the 2nd dimension of the matrix, returning a vector of length of the 1st dimension (the output grid)
area(regridder::AbstractMatrix, dims::Val{:in}) = vec(sum(regridder, dims=2))        
area(regridder::AbstractMatrix, dims::Val{:out}) = vec(sum(regridder, dims=1))      # "out" vice versa


"""
    area_of_intersection_operator([m::Manifold], grid1, grid2; threaded = false, ...)

Compute the sparse matrix operator for the area of intersection.
"""
area_of_intersection_operator(grid1, grid2; threaded = GO.False(), kwargs...) = area_of_intersection_operator(GO.Planar(), grid1, grid2, GO.booltype(threaded))
area_of_intersection_operator(m::GO.Manifold, grid1, grid2; threaded = GO.False(), kwargs...) = area_of_intersection_operator(m, grid1, grid2, GO.booltype(threaded))
area_of_intersection_operator(m::GO.Manifold, grid1, grid2, threaded; kwargs...) = area_of_intersection_operator!(m, spzeros(Float32, length(grid1), length(grid2)), grid1, grid2, GO.booltype(threaded); kwargs...)

"""
    area_of_intersection_operator!([m::Manifold], operator::AbstractMatrix, grid1, grid2; threaded = false, ...)

Write into `operator` the areas of intersection between polygons in `grid1` (dimension 1) and `grid2` (dimension 2).
Assumes `operator` is initialized at zero, but does not check for that.

`operator` may be any AbstractMatrix but should usually be a `SparseMatrixCSC`, even for small grids, due to the storage efficiency and speedup.
"""
area_of_intersection_operator!(operator::AbstractMatrix, grid1, grid2; threaded = GO.False(), kwargs...) = area_of_intersection_operator!(GO.Planar(), operator, grid1, grid2, GO.booltype(threaded); kwargs...)
area_of_intersection_operator!(m::GO.Manifold, operator::AbstractMatrix, grid1, grid2; threaded = GO.False(), kwargs...) = area_of_intersection_operator!(m, operator, grid1, grid2, GO.booltype(threaded); kwargs...)

# Single threaded approach
function area_of_intersection_operator!(m::GO.Manifold, operator::AbstractMatrix, grid1, grid2, threaded::GO.False; nodecapacity = 10)
    # Prepare STRtrees for the two grids, to speed up intersection queries
    # we may want to separately tune nodecapacity if one is much larger than the other.  
    # specifically we may want to tune leaf node capacity via Hilbert packing while still 
    # constraining inner node capacity.  But that can come later.
    tree1 = GO.SortTileRecursiveTree.STRtree(grid1; nodecapacity = first(nodecapacity)) 
    tree2 = GO.SortTileRecursiveTree.STRtree(grid2; nodecapacity = last(nodecapacity))
    # Do the dual query, which is the most efficient way to do this,
    # by iterating down both trees simultaneously, rejecting pairs of nodes that do not intersect.
    # when we find an intersection, we calculate the area of the intersection and add it to the result matrix.
    GO.SpatialTreeInterface.do_dual_query(GO.Extents.intersects, tree1, tree2) do i1, i2
        p1, p2 = grid1[i1], grid2[i2]
        # may want to check if the polygons intersect first, 
        # to avoid antimeridian-crossing multipolygons viewing a scanline.

        if GO.disjoint(p1, p2) # fast rejection path
            return GO.LoopStateMachine.Continue() # return LoopStateMachine.jl no-op
        end

        intersection_polys = try # can remove this now, got all the errors cleared up in the fix.
            # At some future point, we may want to add the manifold here
            # but for right now, GeometryOps only supports planar polygons anyway.
            GO.intersection(p1, p2; target = GI.PolygonTrait())
        catch e
            @error "Intersection failed!" i1 i2
            rethrow(e)
        end

        area_of_intersection = GO.area(#=TODO: manifold support=#intersection_polys)
        if area_of_intersection > 0
            operator[i1, i2] += area_of_intersection
        end
    end

    return operator
end

# Multithreaded approach - still very slow
function area_of_intersection_operator!(m::GO.Manifold, operator::AbstractMatrix, grid1, grid2, threaded::GO.True; nodecapacity = 10
    throw(ArgumentError("Not implemented yet for threaded = true"))
    # all code below is unreachable and not executed for now

    # Prepare STRtrees for the two grids, to speed up intersection queries
    # we may want to separately tune nodecapacity if one is much larger than the other.  
    # specifically we may want to tune leaf node capacity via Hilbert packing while still 
    # constraining inner node capacity.  But that can come later.
    tree1 = GO.SortTileRecursiveTree.STRtree(grid1; nodecapacity = first(nodecapacity)) 
    tree2 = GO.SortTileRecursiveTree.STRtree(grid2; nodecapacity = last(nodecapacity))
    # Do the dual query, which is the most efficient way to do this,
    # by iterating down both trees simultaneously, rejecting pairs of nodes that do not intersect.
    # when we find an intersection, we push that pair to the list of potential intersections.
    # This dual tree query is fast enough that we can run it single-threaded.
    likely_pairs = Vector{Tuple{Int, Int}}()
    GO.SpatialTreeInterface.do_dual_query(GO.Extents.intersects, tree1, tree2) do i1, i2
        p1, p2 = grid1[i1], grid2[i2]
        if GO.intersects(p1, p2)
            push!(likely_pairs, (i1, i2))
        end
    end

    likely_areas = OhMyThreads.tmap(eachindex(likely_pairs)) do i
        i1, i2 = likely_pairs[i]
        intersection_polys = try
            GO.intersection(grid1[i1], grid2[i2]; target = GI.PolygonTrait())
        catch e
            @error "Intersection failed!" i1 i2
            rethrow(e)
        end
        return Float32(GO.area(intersection_polys)) # Compute as Float64, downsize to f32 for operator
    end

    for ((i1, i2), a) in zip(likely_pairs, likely_areas)
        iszero(a) && continue
        operator[i1, i2] += a
    end

    return operator
end
