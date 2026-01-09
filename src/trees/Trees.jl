module Trees

using DocStringExtensions
import LinearAlgebra
import GeoInterface as GI
import GeometryOps as GO
import GeometryOps: UnitSpherical as US, SpatialTreeInterface as STI

include("wrappers.jl")
include("quadtrees.jl")
include("quadtree_cursor.jl")

export AbstractQuadtree, ncells, getcell
export ExplicitPolygonQuadtree, CellBasedQuadtree
export QuadtreeCursor

end