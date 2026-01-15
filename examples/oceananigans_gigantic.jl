using Oceananigans
using ConservativeRegridding
using ConservativeRegridding.Trees
import GeoInterface as GI
import GeometryOps as GO
import GeometryOps: SpatialTreeInterface as STI
using Statistics

include("oceananigans_common.jl")

src_grid = LatitudeLongitudeGrid(size=(2880, 1440, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))
dst_grid = LatitudeLongitudeGrid(size=(2880, 1440, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))
# dst_grid = TripolarGrid(size=(360, 180, 1), fold_topology = RightFaceFolded)


src_field = CenterField(src_grid)
dst_field = CenterField(dst_grid)
set!(src_field, (lon, lat, z) -> lon)

src_longlat_cells = compute_cell_matrix(src_field)
dst_longlat_cells = compute_cell_matrix(dst_field)

src_cells = GO.UnitSphereFromGeographic().(src_longlat_cells)
dst_cells = GO.UnitSphereFromGeographic().(dst_longlat_cells)


src_qt = Trees.CellBasedGrid(src_cells) 
dst_qt = Trees.CellBasedGrid(dst_cells) 

src_tree = src_qt |> Trees.TopDownQuadtreeCursor |> Trees.KnownFullSphereExtentWrapper
dst_tree = dst_qt |> Trees.TopDownQuadtreeCursor |> Trees.KnownFullSphereExtentWrapper

idxs = NTuple{2, Int}[]
@time STI.dual_depth_first_search(GO.UnitSpherical._intersects, src_tree, dst_tree) do i1, i2
    push!(idxs, (i1, i2))
end
idxs

unique(first.(idxs))

# idxs = [Tuple{Int, Int}[] for _ in CartesianIndices(size(src_cells).-1)]
# @time STI.dual_depth_first_search(GO.UnitSpherical._intersects, src_tree, dst_tree) do i1, i2
#     push!(idxs[i1...], (i2))
# end
# idxs

# Sort `idxs` by the first element of the tuple, i.e. the source cell index
# This allows us to then partition it by source cell index so that no computations
# "overlap"...
# sorted_idxs = sort(idxs, by = first)
npartitions = Threads.nthreads()

import ChunkSplitters, ProgressMeter
partitions = ChunkSplitters.chunks(idxs; n = npartitions * 4)
progress = ProgressMeter.Progress(length(partitions); desc = "Computing intersection areas")

function compute_intersection_areas(idxs, progress = nothing)
    ret_i1 = Int[]
    ret_i2 = Int[]
    ret_area = Float64[]
    sizehint!(ret_i1, length(idxs))
    sizehint!(ret_i2, length(idxs))
    sizehint!(ret_area, length(idxs))

    for (i1, i2) in idxs
        p1 = Trees.getcell(src_qt, i1)
        p2 = Trees.getcell(dst_qt, i2)
        polygon_of_intersection = #=try; =#
            GO.intersection(GO.ConvexConvexSutherlandHodgman(GO.Spherical()), p1, p2; target = GO.PolygonTrait()) 
        # catch e; 
        #     # @show "Error during intersection" i1 i2; 
        #     push!(failing_pairs, (i1, i2))
        #     continue
        # end
        area_of_intersection = GO.area(GO.Spherical(), polygon_of_intersection)
        if area_of_intersection > 0
            push!(ret_i1, i1)
            push!(ret_i2, i2)
            push!(ret_area, area_of_intersection)
        end
    end
    if !isnothing(progress)
        ProgressMeter.next!(progress)
    end
    return ret_i1, ret_i2, ret_area
end

result_tasks = [Threads.@spawn compute_intersection_areas(partition, $progress) for partition in partitions]
@time results = @showprogress map(fetch, result_tasks)



ret_i1 = reduce(vcat, getindex.(results, 1))
ret_i2 = reduce(vcat, getindex.(results, 2))
ret_area = reduce(vcat, getindex.(results, 3))
mat = sparse(ret_i1, ret_i2, ret_area, prod(size(src_cells).-1), prod(size(dst_cells).-1))

regridder = ConservativeRegridding.Regridder(
    transpose(mat),
    ConservativeRegridding.areas(GO.Spherical(), dst_tree),
    ConservativeRegridding.areas(GO.Spherical(), src_tree),
    zeros(prod((Trees.ncells(dst_tree, 1), Trees.ncells(dst_tree, 2)))),
    zeros(prod((Trees.ncells(src_tree, 1), Trees.ncells(src_tree, 2))))
)

ConservativeRegridding.regrid!(vec(interior(dst_field)), regridder, vec(interior(src_field)))

heatmap(interior(dst_field, :, :, 1))

areas_dst = ConservativeRegridding.areas(GO.Spherical(), dst_tree)
areas_src = ConservativeRegridding.areas(GO.Spherical(), src_tree)

sum(areas_dst)
sum(areas_src)
sum(mat)

@test all(dst_field .≈ src_field)

@test vec(sum(mat; dims = 1)) ≈ areas_dst
@test sum(mat; dims = 2) ≈ areas_src
