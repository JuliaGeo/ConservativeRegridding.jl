module Trees

using DocStringExtensions
import LinearAlgebra
import GeoInterface as GI
import GeometryOps as GO
import GeometryOps: UnitSpherical as US, SpatialTreeInterface as STI

include("interfaces.jl")
include("neighbours_interface.jl")
include("wrappers.jl")
include("grids.jl")

include("quadtree_cursors.jl")
include("specialized_quadtree_cursors.jl")
include("neighbours.jl")

# private IndexOffsetQuadtreeCursor, CubedSphereToplevelTree, CubeFaceConnectivity

export AbstractCurvilinearGrid, ncells, getcell
export ExplicitPolygonGrid, CellBasedGrid, RegularGrid
export QuadtreeCursor, TopDownQuadtreeCursor
export LonLatConnectivityWrapper
export should_parallelize, WithParallelizePolicy

end