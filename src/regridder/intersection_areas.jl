import GeometryOps as GO
using GeometryOps: SpatialTreeInterface as STI

function compute_intersection_areas(
        manifold::M, 
        intersection_operator::F,
        dst_tree::D, 
        src_tree::S, 
        idxs::AbstractVector{Tuple{Int, Int}}, 
    ) where {M <: Manifold, F, D, S}

    ret_i1 = Int[]
    ret_i2 = Int[]
    ret_area = Float64[]
    sizehint!(ret_i1, length(idxs))
    sizehint!(ret_i2, length(idxs))
    sizehint!(ret_area, length(idxs))

    # Loop through all candidate pairs and attempt intersection.
    for (i1, i2) in idxs
        p1 = Trees.getcell(src_tree, i1)
        p2 = Trees.getcell(dst_tree, i2)
        area_of_intersection = intersection_operator(p1, p2)
        if area_of_intersection > 0
            push!(ret_i1, i1)
            push!(ret_i2, i2)
            push!(ret_area, area_of_intersection)
        end
    end

    return ret_i1, ret_i2, ret_area
end

_area_criterion(cap::GO.UnitSpherical.SphericalCap) = (2pi * (1-cos(cap.radius))) < pi # 60 degree cap - easy enough.  generates quite a few threads anyway.
_area_criterion(cap::Extents.Extent) = error("Area criterion for multithreading has to be customizable, for 2D planar extents - this needs to be implemented!")

function get_all_candidate_pairs(threaded::True, predicate_f::F, src_tree::T1, dst_tree::T2) where {F, T1, T2}
    # TODO: Threaded dual dfs via chunking.
    # For now this is just serial, and is the big bottleneck for larger grids.
    # First, run the dual depth first search to get all candidate pairs of 
    # cells that may intersect.
    candidate_idxs = multithreaded_dual_query(predicate_f, _area_criterion, src_tree, dst_tree) # from utils/MultithreadedDualDepthFirstSearch.jl
    return candidate_idxs
end

function get_all_candidate_pairs(threaded::False, predicate_f::F, src_tree::T1, dst_tree::T2) where {F, T1, T2}
    candidate_idxs = Tuple{Int, Int}[]
    STI.dual_depth_first_search(predicate_f, src_tree, dst_tree) do i1, i2
        push!(candidate_idxs, (i1, i2))
    end
    return candidate_idxs
end

function intersection_areas(
        manifold::M, threaded::True, dst_tree::T1, src_tree::T2;
        intersection_operator::F = DefaultIntersectionOperator(manifold),
        npartitions::Int = Threads.nthreads() * 4,
        progress = false,
    ) where {M <: Manifold,F, T1, T2}

    predicate_f = if M <: Spherical
        GO.UnitSpherical._intersects
    else
        Extents.intersects
    end
    # TODO: Threaded dual dfs via chunking.
    # For now this is just serial, and is the big bottleneck for larger grids.
    # First, run the dual depth first search to get all candidate pairs of 
    # cells that may intersect.
    candidate_idxs = get_all_candidate_pairs(threaded, predicate_f, src_tree, dst_tree)

    # Then, partition the list of indices to check,
    partitions = ChunkSplitters.chunks(candidate_idxs; n = npartitions)
    if progress
        progress_meter = ProgressMeter.Progress(length(partitions); desc = "Computing intersection areas")
    end
    # and compute the intersection areas for each partition in parallel.
    # This is a bit oversubscribed though I guess.  But Julia's dynamic
    # scheduler should handle it fine.
    result_tasks = [
        StableTasks.@spawn begin
            ret = compute_intersection_areas(
                $manifold, 
                $intersection_operator,
                $dst_tree, 
                $src_tree, 
                partition, 
            ) 
            $(progress ? :(ProgressMeter.next!(progress_meter)) : :())
            ret
        end
        for partition in partitions
    ]
    
    # Fetch the results of `result_tasks`
    all_results = map(fetch, result_tasks)
    # Concatenate the results into single vectors
    i1s = reduce(vcat, getindex.(all_results, 1))
    i2s = reduce(vcat, getindex.(all_results, 2))
    areas = reduce(vcat, getindex.(all_results, 3))
    # Assemble a sparse matrix from the results.
    return SparseArrays.sparse(
        i2s, 
        i1s, 
        areas, 
        prod(Trees.ncells(dst_tree)), 
        prod(Trees.ncells(src_tree)),
    )
end


function intersection_areas(
        manifold::M, threaded::False, dst_tree::T1, src_tree::T2;
        intersection_operator::F = DefaultIntersectionOperator(manifold),
        npartitions::Int = Threads.nthreads() * 4,
        progress = false,
    ) where {M <: Manifold,F, T1, T2}

    predicate_f = if M <: Spherical
        GO.UnitSpherical._intersects
    else
        Extents.intersects
    end

    candidate_idxs = get_all_candidate_pairs(threaded, predicate_f, src_tree, dst_tree)

    # if progress
    #     progress_meter = ProgressMeter.Progress(length(candidate_idxs); desc = "Computing intersection areas")
    # end
    i1s, i2s, areas = compute_intersection_areas(
        manifold, 
        intersection_operator,
        dst_tree, 
        src_tree, 
        candidate_idxs
    ) 
    # Assemble a sparse matrix from the results.
    return SparseArrays.sparse(
        i2s, 
        i1s, 
        areas, 
        prod(Trees.ncells(dst_tree)), 
        prod(Trees.ncells(src_tree)),
    )
end