using Rasters, RasterDataSources, ArchGDAL
import DimensionalData as DD

import GeoInterface as GI, GeometryOps as GO
import StaticArrays

bigras = Raster(WorldClim{Climate}, :tavg; month = 6)
ras = Rasters.aggregate(sum, bigras, 3)
xb = X(DD.intervalbounds(ras, X))
yb = Y(DD.intervalbounds(ras, Y))

function _rectfrombounds((xmin, xmax), (ymin, ymax))
    GI.Polygon(StaticArrays.@SVector[GI.LinearRing(StaticArrays.@SVector[(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)])])
end
using Extents
function _extentfrombounds(X, Y)
    Extents.Extent(; X, Y)
end

rects = @d _rectfrombounds.(xb, yb)
exts = @d _extentfrombounds.(xb, yb)

# here comes the work
using NaturalEarth
all_countries = naturalearth("admin_0_countries", 10)

import ConservativeRegridding


intersections = @time ConservativeRegridding.intersection_areas(exts, all_countries.geometry; threaded = true, area_of_intersection_operator = (e, p) -> GO.coverage(p, e))

# If the two grids completely overlap, then the areas should be equivalent
# to the sum of the intersection areas along the second and fisrt dimensions, 
# for src and dst, respectively. This is not the case if the two grids do not cover the same area.
dst_areas = GeometryOps.area.(all_countries.geometry) 
src_areas = GeometryOps.area.(src_polys) 

regridder = ConservativeRegridding.Regridder(intersections, dst_areas, src_areas)
normalize && LinearAlgebra.normalize!(regridder)