module ConservativeRegridding

using DocStringExtensions
import LinearAlgebra
import GeoInterface
import GeometryOps
import SortTileRecursiveTree
import Extents
import SparseArrays

# piracy, remove when GeometryOps is fixed
# GeometryOps.area(::GeometryOps.Planar, x) = GeometryOps.area(x)

import GeometryOps as GO
import GeoInterface as GI

using SparseArrays: spzeros # for default operator construction

include("regridder.jl")
include("regrid.jl")

end
