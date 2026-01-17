using Oceananigans
using ConservativeRegridding
using ConservativeRegridding.Trees
import GeoInterface as GI
import GeometryOps as GO
import GeometryOpsCore as GOCore
import GeometryOps: SpatialTreeInterface as STI
using Statistics
using GLMakie, GeoMakie
using ClimaOcean
import LibGEOS

include("oceananigans_common.jl")

src_field = JRA55FieldTimeSeries(:temperature, CPU(), Float64)[1]
src_grid  = src_field.grid

dst_grid = ConformalCubedSphereGrid(panel_size=(100, 100, 1), z = (0, 1))
dst_field = CenterField(dst_grid)

@time @apply_regionally regridder = ConservativeRegridding.Regridder(
    dst_field,
    src_field
)

@apply_regionally dst = vec(interior(dst_field))
@time @apply_regionally ConservativeRegridding.regrid!(dst, regridder, vec(interior(src_field)))

fig = Figure()
ax = GlobeAxis(fig[1, 1])
cmin = minimum(minimum, (interior(dst) for dst in (dst_field[i] for i in 1:6)))
cmax = maximum(maximum, (interior(dst) for dst in (dst_field[i] for i in 1:6)))
for dst in (dst_field[i] for i in 1:6)
    poly!(
        ax, 
        GI.convert.((LibGEOS,), vec(GO.transform(GO.UnitSpherical.GeographicFromUnitSphere(), Trees.getcell(Trees.treeify(dst))))); 
        color = vec(interior(dst)), 
        colorrange = (-30 + 273, 30 + 273),
        colormap = :thermal
    )
end

@apply_regionally dst_trees = Trees.treeify(dst_field)
@apply_regionally areas_dst = ConservativeRegridding.areas(GO.Spherical(), dst_trees)
areas_src = ConservativeRegridding.areas(GO.Spherical(), Trees.treeify(src_field))
dst_sum   = sum(sum(areas_dst[i] .* dst[i]) for i in 1:6)
src_sum   = sum(areas_src .* vec(interior(src_field)))

# Test conservation!

@show (sum_src - sum_dst) / sum_src
