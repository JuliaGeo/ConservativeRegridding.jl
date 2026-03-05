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

include("regridder/regridder.jl")
include("regridder/regrid.jl")
include("regridder/intersection_areas.jl")

"""
    VelocityLineIntegralRegridder(args...; kwargs...)

Construct a velocity transport remapper.

The concrete implementation is provided by package extensions (currently the
Oceananigans extension). If the required extension is not loaded, this fallback
throws an informative error.
"""
function VelocityLineIntegralRegridder(args...; kwargs...)
    throw(ArgumentError("VelocityLineIntegralRegridder requires an extension implementation (for Oceananigans, load Oceananigans + LibGEOS)."))
end

"""
    regrid_velocity_transport!(dst_u, dst_v, R, src_u, src_v)

Apply an extension-provided velocity transport remap operator.
"""
function regrid_velocity_transport!(dst_u, dst_v, R, src_u, src_v)
    throw(ArgumentError("regrid_velocity_transport! requires an extension implementation (for Oceananigans, load Oceananigans + LibGEOS)."))
end

@public Regridder, regrid, regrid!
@public VelocityLineIntegralRegridder, regrid_velocity_transport!
@public areas

end
