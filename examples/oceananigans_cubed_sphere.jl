using Oceananigans
using ConservativeRegridding
using ConservativeRegridding: Trees

cubed_sphere_grid = ConformalCubedSphereGrid(panel_size=(50, 50, 1), z=(-1, 0))
cubed_sphere_field = CenterField(cubed_sphere_grid)

longlat_grid = LatitudeLongitudeGrid(size=(360, 180, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))
longlat_field = CenterField(longlat_grid)

src_field = longlat_field
dst_field = cubed_sphere_field

regridders = [
    ConservativeRegridding.Regridder(dst_field, src_field)
    for dst_field in (cubed_sphere_field[i] for i in 1:6)
]

set!(src_field, VortexField(; lat0_rad = deg2rad(80)))

for (regridder, dst) in zip(regridders, (cubed_sphere_field[i] for i in 1:6))
    ConservativeRegridding.regrid!(vec(interior(dst)), regridder, vec(interior(src_field)))
end


using GeoMakie, GLMakie
fig = Figure()
ax = GlobeAxis(fig[1, 1])
for dst in (cubed_sphere_field[i] for i in 1:6)
    poly!(ax, GI.convert.((LibGEOS,), vec(GO.transform(GO.UnitSpherical.GeographicFromUnitSphere(), Trees.getcell(Trees.treeify(dst))))); color = vec(interior(dst)))
end
1

