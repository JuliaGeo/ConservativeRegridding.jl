module ConservativeRegridding

using DocStringExtensions
import LinearAlgebra
import GeoInterface
import GeometryOps
import SortTileRecursiveTree
import Extents

import GeometryOps as GO
import GeoInterface as GI

include("regridder.jl")
include("regrid.jl")

end
