using ConservativeRegridding
using ConservativeRegridding.Trees

import GeometryOps as GO, GeometryOpsCore as GOCore
import GeoInterface as GI

# Import Healpix as a regular Julia package
import Healpix
using Oceananigans

using Test

lonlat_fine_grid = LatitudeLongitudeGrid(size=(360, 180, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))
lonlat_fine_field = CenterField(lonlat_fine_grid)

set!(lonlat_fine_field, VortexField(; lat0_rad = deg2rad(80)))

# Can handle both orders, nested and ring order
healpix_fine_grid = Healpix.HealpixMap{Float64, Healpix.NestedOrder}(64)
healpix_fine_grid = Healpix.HealpixMap{Float64, Healpix.RingOrder}(64)
healpix_fine_field = healpix_fine_grid.pixels


regridder = ConservativeRegridding.Regridder(GO.Spherical(), healpix_fine_grid, lonlat_fine_field; normalize = false)
ConservativeRegridding.regrid!(vec(healpix_fine_field), regridder, vec(interior(lonlat_fine_field)))

using GLMakie, GeoMakie
f, a, p = poly(GO.transform(GO.UnitSpherical.GeographicFromUnitSphere(), Trees.getcell(Trees.treeify(healpix_fine_grid))) |> vec; color = vec(healpix_fine_field), axis = (; type = GlobeAxis))
lines!(a, GeoMakie.coastlines(); zlevel = 100_000, color = :orange)
f

# Test areas are correct
areas_lonlat = ConservativeRegridding.areas(GO.Spherical(), Trees.treeify(lonlat_fine_grid))
areas_healpix = ConservativeRegridding.areas(GO.Spherical(), Trees.treeify(healpix_fine_grid))

@test sum(regridder.intersections, dims = 2)[:, 1] ≈ areas_healpix
@test sum(regridder.intersections, dims = 1)[1, :] ≈ areas_lonlat

# Test integral was conserved
@test sum(vec(healpix_fine_field) .* areas_healpix) ≈ sum(vec(interior(lonlat_fine_field)) .* areas_lonlat) rtol=1e-15