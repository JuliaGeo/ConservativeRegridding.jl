#=
# SpeedyWeather.jl grid regridding

We support certain grids from SpeedyWeather/RingGrids.jl directly.

For now, these are only the full grids, not the healpix or reduced grids.  So `Full`
=#
using SpeedyWeather
using ConservativeRegridding

src = rand(FullClenshawGrid, 24)
dst = rand(FullClenshawGrid, 48)
#
R = @time ConservativeRegridding.Regridder(dst, src)
# Given this regridder, we can now regrid from src to dst and back.
ConservativeRegridding.regrid!(dst, R, src)

# The reverse direction shares the same sparse matrix (via transpose) and
# needs no new construction.
ConservativeRegridding.regrid!(src, transpose(R), dst)
