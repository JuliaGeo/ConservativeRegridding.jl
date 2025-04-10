module ConservativeRegridding

using DocStringExtensions
import LinearAlgebra
import GeoInterface
import GeometryOps
import SortTileRecursiveTree
import Extents
import SparseArrays

# piracy, remove when GeometryOps is fixed
GeometryOps.area(::GeometryOps.Planar, x) = GeometryOps.area(x)

include("regridder.jl")
include("regrid.jl")

end
