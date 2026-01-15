import GeometryOps as GO
import GeoInterface as GI
using GeometryOps: SpatialTreeInterface as STI
using GeometryOps.LoopStateMachine: @controlflow
using StableTasks
using Test


# Walk through both trees and kick off a task whenever you hit a node that satisfies the area criterion.
function multithreaded_dfs(inner_dfs_f::IF, predicate::P, area_criterion::A, tasks::V, node1::N1, node2::N2) where {P, IF, A, V <: Vector{<: StableTasks.StableTask}, N1, N2}
    if STI.isleaf(node1) || STI.isleaf(node2)
        # both nodes are leaves, so we can just iterate over the indices and extents
        push!(tasks, StableTasks.@spawn $inner_dfs_f($predicate, node1, node2))
    else
        # neither node is a leaf, recurse into both children
        if area_criterion(STI.node_extent(node1)) && area_criterion(STI.node_extent(node2))
            push!(tasks, StableTasks.@spawn $inner_dfs_f($predicate, node1, node2))
        else
            for child1 in STI.getchild(node1)
                for child2 in STI.getchild(node2)
                    if predicate(STI.node_extent(child1), STI.node_extent(child2))
                        @controlflow multithreaded_dfs(inner_dfs_f, predicate, area_criterion, tasks, child1, child2)
                    end
                end
            end
        end
    end
    return tasks
end

using Oceananigans
using ConservativeRegridding
using ConservativeRegridding.Trees

src_grid = LatitudeLongitudeGrid(size=(100, 100, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))
dst_grid = LatitudeLongitudeGrid(size=(100, 100, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))

src_field = CenterField(src_grid)
dst_field = CenterField(dst_grid)
set!(src_field, (lon, lat, z) -> lon)

src_tree = Trees.treeify(src_field)
dst_tree = Trees.treeify(dst_field)

function area_of_spherical_cap(cap)
    return 2pi * (1-cos(cap.radius))
end
function area_c(cap)
    area_of_spherical_cap(cap) <= 0.5892243899248773
end


function inner_f(predicate::P, node1::N1, node2::N2) where {P, N1, N2}
    ret = Tuple{Int, Int}[]
    STI.dual_depth_first_search(predicate, node1, node2) do i1, i2
        push!(ret, (i1, i2))
    end
    return ret
end

tasks = StableTasks.StableTask{Vector{Tuple{Int, Int}}}[]
# tasks = StableTasks.StableTask{Any}[]
multithreaded_dfs(inner_f, GO.UnitSpherical._intersects, area_c, tasks, src_tree, dst_tree)

results = map(fetch, tasks)
result = reduce(vcat, results)

function do_multithreaded_dfs(inner_dfs_f::IF, predicate::P, area_criterion::A, node1::N1, node2::N2) where {IF, P, A, N1, N2}
    tasks = StableTasks.StableTask{Vector{Tuple{Int, Int}}}[]
    multithreaded_dfs(inner_dfs_f, predicate, area_criterion, tasks, node1, node2)
    return reduce(vcat, map(fetch, tasks))
end

@time result = do_multithreaded_dfs(inner_f, GO.UnitSpherical._intersects, area_c, src_tree, dst_tree)
@time result = inner_f(GO.UnitSpherical._intersects, src_tree, dst_tree)

@test isempty(setdiff(result, inner_f(GO.UnitSpherical._intersects, src_tree, dst_tree)))

inner_f(GO.UnitSpherical._intersects, src_tree, dst_tree)