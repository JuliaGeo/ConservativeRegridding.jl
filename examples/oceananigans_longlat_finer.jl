using Oceananigans
using ConservativeRegridding
using ConservativeRegridding.Trees
import GeoInterface as GI
import GeometryOps as GO
import GeometryOps: SpatialTreeInterface as STI
using Statistics

include("oceananigans_common.jl")

longlat_coarse_grid = LatitudeLongitudeGrid(size=(100, 100, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))
longlat_fine_grid = LatitudeLongitudeGrid(size=(200, 200, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))
tripolar_grid = TripolarGrid(size=(360, 180, 1), fold_topology = RightFaceFolded)

src_grid = longlat_coarse_grid
dst_grid = longlat_fine_grid

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

cartesian_idxs = map(idxs) do (i1, i2)
    idx1 = Trees.linear_to_cartesian_idx(src_qt, i1)
    idx2 = Trees.linear_to_cartesian_idx(dst_qt, i2)
    (idx1, idx2)
end

src_to_dst_map = [CartesianIndex{2}[] for i in 1:Trees.ncells(src_qt, 1), j in 1:Trees.ncells(src_qt, 2)]
for (i1, i2) in cartesian_idxs
    push!(src_to_dst_map[i1], i2)
end

is_fine = [
    begin
        src_i = src_idx[1]
        src_j = src_idx[2]
        dst_i = src_i * 2 - 1
        dst_j = src_j * 2 - 1
        dst_idxs = CartesianIndex.(((dst_i, dst_j), (dst_i+1, dst_j+1), (dst_i, dst_j+1), (dst_i+1, dst_j)))
        all(in.(dst_idxs, (src_to_dst_map[src_idx],)))
    end
    for src_idx in CartesianIndices(src_to_dst_map)
]
@test all(is_fine)

all_dst_polys = [
    begin
        src_i = src_idx[1]
        src_j = src_idx[2]
        dst_i = src_i * 2 - 1
        dst_j = src_j * 2 - 1
        dst_idxs = CartesianIndex.(((dst_i, dst_j), (dst_i+1, dst_j+1), (dst_i, dst_j+1), (dst_i+1, dst_j)))

        src_poly = Trees.getcell(src_qt, src_idx)

        dst_polys = Trees.getcell.((dst_qt,), dst_idxs)
    end
    for src_idx in CartesianIndices(src_to_dst_map)
]

multipolys = reduce(vcat, [[GI.MultiPolygon([Trees.getcell(src_qt, idx), p]) for p in polys] for (idx, polys) in enumerate(all_dst_polys)])
multipolys_longlat = GO.transform(GO.GeographicFromUnitSphere(), multipolys)
GeoJSON.write("multipolys.geojson", [(; geometry = mp) for mp in multipolys_longlat])

src_idx = CartesianIndex(1, 1)

src_i = src_idx[1]
src_j = src_idx[2]
dst_i = src_i * 2 - 1
dst_j = src_j * 2 - 1
dst_idxs = CartesianIndex.(((dst_i, dst_j), (dst_i+1, dst_j+1), (dst_i, dst_j+1), (dst_i+1, dst_j)))

src_poly = Trees.getcell(src_qt, src_idx)

dst_polys = Trees.getcell.((dst_qt,), dst_idxs)

intersection_areas = ntuple(4) do i
    GO.area(
        GO.Spherical(),
        GO.intersection(GO.ConvexConvexSutherlandHodgman(GO.Spherical()), src_poly, dst_polys[i]; target = GO.PolygonTrait())
    )
end

import LibGEOS

src_poly_longlat = GO.transform(GO.GeographicFromUnitSphere(), src_poly)
dst_polys_longlat = GO.transform(GO.GeographicFromUnitSphere(), dst_polys; threaded = false)

fig, ax, plt = poly(GI.convert(LibGEOS, src_poly_longlat); transparency = true, strokewidth = 1)
plt2 = poly!(ax, GI.convert(LibGEOS, dst_polys_longlat[1]); transparency = true, strokewidth = 1)
poly!(ax, GI.convert(LibGEOS, polygon_of_intersection_longlat); color = :red, transparency = true, strokewidth = 1)
polygon_of_intersection = GO.intersection(GO.ConvexConvexSutherlandHodgman(GO.Spherical()), src_poly, dst_polys[1]; target = GO.PolygonTrait())

polygon_of_intersection_longlat = GO.transform(GO.GeographicFromUnitSphere(), polygon_of_intersection)

using GeoJSON
GeoJSON.write("bad_longlat_polys.geojson", [(; geometry = src_poly_longlat, name = "p1"), (geometry = dst_polys_longlat[1], name = "p2")])

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
pairs = [Tuple{Int, Int, Float64}[] for i in 1:npartitions]

# partitions = 

using SparseArrays, ProgressMeter

mat = spzeros(Float64, prod(size(src_cells).-1), prod(size(dst_cells).-1))

failing_pairs = Tuple{Int, Int}[]
@time @showprogress for (i1, i2) in idxs
    p1 = Trees.getcell(src_qt, i1)
    p2 = Trees.getcell(dst_qt, i2)
    polygon_of_intersection = try;
        GO.intersection(GO.ConvexConvexSutherlandHodgman(GO.Spherical()), p1, p2; target = GO.PolygonTrait()) 
    catch e; 
        # @show "Error during intersection" i1 i2; 
        push!(failing_pairs, (i1, i2))
        continue
    end
    area_of_intersection = GO.area(GO.Spherical(), polygon_of_intersection)
    mat[i1, i2] += area_of_intersection
end
failing_pairs
mat

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

vec(sum(mat; dims = 1)) |> sum
vec(sum(mat; dims = 2)) |> sum

@test all(sum((dst_field * Oceananigans.Operators.Az)) ≈ sum((src_field * Oceananigans.Operators.Az)))

sum(dst_field * Oceananigans.Operators.Az)
sum(src_field * Oceananigans.Operators.Az)

@test all(vec(sum(mat; dims = 1)) .≈ areas_dst)
@test all(vec(sum(mat; dims = 2)) .≈ areas_src)
