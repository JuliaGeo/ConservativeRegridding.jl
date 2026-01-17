module MultithreadedDualDepthFirstSearch

using GeometryOps: SpatialTreeInterface as STI
using GeometryOps.LoopStateMachine: @controlflow
import ProgressMeter
import StableTasks

# Walk through both trees and kick off a task whenever you hit a node that satisfies the area criterion.
function multithreaded_dual_depth_first_search(inner_dfs_f::IF, predicate::P, area_criterion::A, tasks::V, node1::N1, node2::N2) where {P, IF, A, V <: Vector{<: StableTasks.StableTask}, N1, N2}
    if STI.isleaf(node1) || STI.isleaf(node2)
        # both nodes are leaves, so we want to run the inner dfs function on the nodes themselves.
        push!(tasks, StableTasks.@spawn $inner_dfs_f($predicate, node1, node2))
    else
        # neither node is a leaf, recurse into both children
        if area_criterion(STI.node_extent(node1)) && area_criterion(STI.node_extent(node2))
            push!(tasks, StableTasks.@spawn $inner_dfs_f($predicate, node1, node2))
        else
            for child1 in STI.getchild(node1)
                for child2 in STI.getchild(node2)
                    if predicate(STI.node_extent(child1), STI.node_extent(child2))
                        @controlflow multithreaded_dual_depth_first_search(inner_dfs_f, predicate, area_criterion, tasks, child1, child2)
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

function multithreaded_dual_query(predicate::P, area_criterion::A, node1::N1, node2::N2; progress = false) where {P, A, N1, N2}
    tasks = StableTasks.StableTask{Vector{Tuple{Int, Int}}}[]
    multithreaded_dual_depth_first_search(_inner_dfs_f, predicate, area_criterion, tasks, node1, node2)
    return reduce(vcat, map(fetch, tasks))
end

export multithreaded_dual_depth_first_search, multithreaded_dual_query
end