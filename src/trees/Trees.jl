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
include("specialized_quadtree_cursors.jl")

# private IndexOffsetQuadtreeCursor, CubedSphereToplevelTree

export AbstractCurvilinearGrid, ncells, cell_index_count, getcell
export ExplicitPolygonGrid, CellBasedGrid, RegularGrid
export QuadtreeCursor, TopDownQuadtreeCursor
export should_parallelize, WithParallelizePolicy
export split_weight

end
