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

using SciMLPublic: @public

include("utils/MultithreadedDualDepthFirstSearch.jl")
using .MultithreadedDualDepthFirstSearch

include("utils/example_data.jl")
export ExampleFieldFunction, LongitudeField, SinusoidField, HarmonicField, GulfStreamField, VortexField

include("trees/Trees.jl")
using .Trees

export AbstractCurvilinearGrid, ncells, getcell
export ExplicitPolygonGrid, CellBasedGrid, RegularGrid
export QuadtreeCursor, TopDownQuadtreeCursor

include("regridder/lagrange.jl")
include("regridder/triangle_quadrature.jl")

include("regridder/regridder.jl")
include("regridder/regrid.jl")
include("regridder/intersection_areas.jl")


@public Regridder, regrid, regrid!
@public AbstractMapping, AbstractNormalizingMapping, FVtoFV, SEtoFV, FVtoSE
@public FVtoFVRegridder, SEtoFVRegridder, FVtoSERegridder
@public destination_areas, source_areas
@public areas

end
