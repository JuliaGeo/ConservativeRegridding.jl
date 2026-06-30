using Rasters, RasterDataSources, NCDatasets
import DimensionalData as DD
using Dates

import GeoInterface as GI, GeometryOps as GO
import SortTileRecursiveTree
import StaticArrays

import NaturalEarth
import ConservativeRegridding
using ConservativeRegridding: Trees

#
get!(ENV, "RASTERDATASOURCES_PATH", pwd())
bigras = Raster(TerraClimate{Historical}, :tmax; date = Date(2000), lazy = true)[Ti(1)]
ras = Rasters.aggregate(sum, bigras, 4)

#

xb = X(DD.intervalbounds(ras, X))
yb = Y(DD.intervalbounds(ras, Y))
(xb, yb)

#

function _rectfrombounds((xmin, xmax), (ymin, ymax))
    GI.Polygon(StaticArrays.@SVector[GI.LinearRing(StaticArrays.@SVector[(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)])])
end
rects = @d _rectfrombounds.(xb, yb)

# here comes the work
all_countries = NaturalEarth.naturalearth("admin_0_countries", 10)

treeify_str(geoms) = Trees.GeometryMaintainingTreeWrapper(geoms, SortTileRecursiveTree.STRtree(geoms))

intersections = @time ConservativeRegridding.intersection_areas(
    GO.Planar(), 
    GO.False(),
    treeify_str(rects), 
    treeify_str(all_countries.geometry[1:2]); 
    intersection_operator = (p1, p2) -> GO.coverage(p1, GI.extent(p2))
)

grid = all_countries.geometry
findfirst(g -> !(GI.trait(g) isa Union{GI.AbstractPolygonTrait, GI.AbstractMultiPolygonTrait}), grid)

# If the two grids completely overlap, then the areas should be equivalent
# to the sum of the intersection areas along the second and fisrt dimensions, 
# for src and dst, respectively. This is not the case if the two grids do not cover the same area.
dst_areas = GeometryOps.area.(all_countries.geometry) 
src_areas = GeometryOps.area.(src_polys) 

regridder = ConservativeRegridding.Regridder(intersections, dst_areas, src_areas)
normalize && LinearAlgebra.normalize!(regridder)