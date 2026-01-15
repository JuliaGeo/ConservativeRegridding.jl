using Oceananigans
using ConservativeRegridding
using ConservativeRegridding.Trees
import GeoInterface as GI
import GeometryOps as GO
import GeometryOps: SpatialTreeInterface as STI
using Statistics

include("oceananigans_common.jl")

src_grid = LatitudeLongitudeGrid(size=(2880, 1440, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))
# dst_grid = LatitudeLongitudeGrid(size=(2880, 1440, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))
dst_grid = TripolarGrid(size=(2880, 1440, 1), fold_topology = RightFaceFolded)

src_field = CenterField(src_grid)
dst_field = CenterField(dst_grid)
set!(src_field, VortexField(; lat0_rad = deg2rad(80)))

@time regridder = ConservativeRegridding.Regridder(dst_field, src_field; threaded = true, normalize = false, progress = true)

@time ConservativeRegridding.regrid!(vec(interior(dst_field)), regridder, vec(interior(src_field)))

using GeoMakie, GLMakie; import LibGEOS

dst_polys = collect(Trees.getcell(Trees.treeify(dst_field))) |>  x-> GO.transform(GO.UnitSpherical.GeographicFromUnitSphere(), x) .|> GI.convert(LibGEOS) |> vec
f, a, p = poly(dst_polys; color = vec(interior(dst_field)), axis = (; type = GlobeAxis))
lines!(a, GeoMakie.coastlines(); zlevel = 100_000, color = :orange)
f