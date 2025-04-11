using Oceananigans
using ConservativeRegridding

global_grid = LatitudeLongitudeGrid(size=(90, 45, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))
hi_res_regional_grid = LatitudeLongitudeGrid(size=(60, 60, 1), longitude=(0, 60), latitude=(0, 60), z=(0, 1))

c1 = CenterField(global_grid)
c2 = CenterField(hi_res_regional_grid)

import GeoInterface as GI, GeometryOps as GO
polys1 = GI.Polygon.(GI.LinearRing.(eachcol(vertices1))) .|> GO.fix
polys2 = GI.Polygon.(GI.LinearRing.(eachcol(vertices2))) .|> GO.fix
regridder = ConservativeRegridding.regridder(polys1, polys2)

ConservativeRegridding.regrid!(c2, regridder, c1)

# semantics: going from b → c
# c = (R * b) ./ ac
#
# b = (Rᵀ * c) ./ ab
#
# ac = sum(R, 2)
# ab = sum(R, 1)
