module ConservativeRegridding

using DocStringExtensions
import LinearAlgebra
import GeoInterface
import GeometryOps
import SortTileRecursiveTree
import Extents
import SparseArrays
import ChunkSplitters
import StableTasks
import ProgressMeter

using GeometryOpsCore: booltype, BoolsAsTypes, True, False, istrue
using GeometryOpsCore: Manifold, Planar, Spherical

include("utils/MultithreadedDualDepthFirstSearch.jl")
using .MultithreadedDualDepthFirstSearch

include("trees/Trees.jl")
using .Trees

export AbstractCurvilinearGrid, ncells, getcell
export ExplicitPolygonGrid, CellBasedGrid, RegularGrid
export QuadtreeCursor, TopDownQuadtreeCursor

include("regridder/regridder.jl")
include("regridder/regrid.jl")
include("regridder/intersection_areas.jl")
include("regridder/adjacency.jl")
include("regridder/gradients.jl")

public Regridder, regrid, regrid!
export AbstractRegridMethod, Conservative1stOrder, Conservative2ndOrder
public areas

end
