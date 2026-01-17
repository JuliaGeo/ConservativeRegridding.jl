module Trees

using DocStringExtensions
import LinearAlgebra
import GeoInterface as GI
import GeometryOps as GO
import GeometryOps: UnitSpherical as US, SpatialTreeInterface as STI

include("interfaces.jl")
include("wrappers.jl")
include("grids.jl")

include("quadtree_cursors.jl")

export AbstractCurvilinearGrid, ncells, getcell
export ExplicitPolygonGrid, CellBasedGrid, RegularGrid
export QuadtreeCursor, TopDownQuadtreeCursor

end