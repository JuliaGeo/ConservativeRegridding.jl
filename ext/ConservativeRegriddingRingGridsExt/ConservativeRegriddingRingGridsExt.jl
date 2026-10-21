module ConservativeRegriddingRingGridsExt

import ConservativeRegridding

using RingGrids

import GeometryOpsCore: best_manifold, manifold, Manifold, Spherical
import GeometryOps as GO
import GeoInterface as GI
import StaticArrays: SA, SVector

import GridInterface as GrI
using GridInterface: UnitSphericalPoint
using GridInterface.Structured
import GridInterface.Structured: nfaces, facesize, facevertex, faceposition, facecell, facecross, facefan,
    facecolumnmajor, hasfaceaxes, faceaxes, facecoords, faceperiod, FaceSeams, seamcross


best_manifold(grid::RingGrids.AbstractGrid) = Spherical()
best_manifold(field::RingGrids.AbstractField) = best_manifold(field.grid)

"""
    ConservativeRegridding.asgrid(grid::RingGrids.AbstractGrid)
    ConservativeRegridding.asgrid(field::RingGrids.AbstractField)

The grid `Regridder` regrids for a RingGrids grid or field: its provider,
[`ringgrid`](@ref).
"""
ConservativeRegridding.asgrid(grid::RingGrids.AbstractGrid) = ringgrid(grid)
ConservativeRegridding.asgrid(field::RingGrids.AbstractField) = ringgrid(field.grid)

# OctaHEALPix closed forms: ring-order <-> (ring, quadrant, index) <-> (r, c) and cell corners.
include("octahealpix.jl")

# Standard 12-face HEALPix closed forms: `xyf2loc`, pixel polygons, ring-order <-> (face, x, y).
include("healpix.jl")

# GridInterface providers over those closed forms (see providers.jl).
include("providers.jl")

end
