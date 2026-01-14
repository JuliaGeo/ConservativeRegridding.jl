module ConservativeRegridding

using DocStringExtensions
import LinearAlgebra
import GeoInterface
import GeometryOps
import SortTileRecursiveTree
import Extents
import SparseArrays
import ProgressMeter
# piracy, remove when GeometryOps is fixed
# GeometryOps.area(::GeometryOps.Planar, x) = GeometryOps.area(x)

include("trees/Trees.jl")
using .Trees

export AbstractCurvilinearGrid, ncells, getcell
export ExplicitPolygonGrid, CellBasedGrid, RegularGrid
export QuadtreeCursor, TopDownQuadtreeCursor

include("regridder/regridder.jl")
include("regridder/regrid.jl")

public Regridder

end
