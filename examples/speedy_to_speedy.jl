# Conservative regridding between two SpeedyWeather FullClenshawGrid
# resolutions. Supported directly via ConservativeRegridding's RingGrids
# extension on `AbstractFullGrid`. Full Gaussian / full equiangular grids work
# through the same path; HEALPix, OctaHEALPix, and reduced Gaussian grids are
# tracked in https://github.com/JuliaGeo/ConservativeRegridding.jl/issues/89.

using SpeedyWeather
using ConservativeRegridding

src = rand(FullClenshawGrid, 24)
dst = rand(FullClenshawGrid, 48)

R = ConservativeRegridding.Regridder(dst, src)

ConservativeRegridding.regrid!(dst, R, src)

# The reverse direction shares the same sparse matrix (via transpose) and
# needs no new construction.
ConservativeRegridding.regrid!(src, transpose(R), dst)
