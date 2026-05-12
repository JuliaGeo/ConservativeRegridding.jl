module MultithreadedDualDepthFirstSearch

using GeometryOps: SpatialTreeInterface as STI
using GeometryOps.LoopStateMachine: @controlflow
import ProgressMeter
import StableTasks

# Walk through both trees and kick off a task whenever both nodes' parallelize
# predicates fire. Each parallelize_k closure is `(node, extent) -> Bool` and is
# pre-bound to its tree by the caller. Extents are computed exactly once per
# node and threaded through the recursion.
function multithreaded_dual_depth_first_search(
    inner_dfs_f::IF, predicate::P,
    parallelize1::A1, parallelize2::A2,
    tasks::V,
    node1::N1, ext1::E1,
    node2::N2, ext2::E2,
) where {IF, P, A1, A2, V <: Vector{<: StableTasks.StableTask}, N1, E1, N2, E2}
    if STI.isleaf(node1) || STI.isleaf(node2)
        # one or both nodes are leaves, so we want to run the inner dfs function on the nodes themselves.
        push!(tasks, StableTasks.@spawn $inner_dfs_f($predicate, node1, node2))
    else
        # neither node is a leaf, recurse into both children
        if parallelize1(node1, ext1) && parallelize2(node2, ext2)
            push!(tasks, StableTasks.@spawn $inner_dfs_f($predicate, node1, node2))
        else
            for child1 in STI.getchild(node1)
                cext1 = STI.node_extent(child1)
                for child2 in STI.getchild(node2)
                    cext2 = STI.node_extent(child2)
                    if predicate(cext1, cext2)
                        @controlflow multithreaded_dual_depth_first_search(
                            inner_dfs_f, predicate, parallelize1, parallelize2, tasks,
                            child1, cext1, child2, cext2,
                        )
                    end
                end
            end
        end
    end
    return tasks
end

function _inner_dfs_f(predicate::P, node1::N1, node2::N2) where {P, N1, N2}
    ret = Tuple{Int, Int}[]
    STI.dual_depth_first_search(predicate, node1, node2) do i1, i2
        push!(ret, (i1, i2))
    end
    return ret
end

function multithreaded_dual_query(
    predicate::P, parallelize1::A1, parallelize2::A2,
    node1::N1, node2::N2;
    progress = false,
) where {P, A1, A2, N1, N2}
    tasks = StableTasks.StableTask{Vector{Tuple{Int, Int}}}[]
    ext1 = STI.node_extent(node1)
    ext2 = STI.node_extent(node2)
    multithreaded_dual_depth_first_search(
        _inner_dfs_f, predicate, parallelize1, parallelize2, tasks,
        node1, ext1, node2, ext2,
    )
    return reduce(vcat, map(fetch, tasks))
end

export multithreaded_dual_depth_first_search, multithreaded_dual_query
end
