using Oceananigans
using Oceananigans.Grids: ξnode, ηnode
using Oceananigans.Fields: AbstractField
using KernelAbstractions: @index, @kernel
using ConservativeRegridding
using ConservativeRegridding.Trees
import GeoInterface as GI
import GeometryOps as GO
import GeometryOps: SpatialTreeInterface as STI
using Statistics

instantiate(L) = L()

function compute_cell_matrix(field::AbstractField)
    Nx, Ny, _ = size(field.grid)
    ℓx, ℓy    = Center(), Center()

    if isnothing(ℓx) || isnothing(ℓy)
        error("cell_matrix can only be computed for fields with non-nothing horizontal location.")
    end

    grid = field.grid
    arch = grid.architecture
    FT = eltype(grid)

    ArrayType = Oceananigans.Architectures.array_type(arch)
    cell_matrix = ArrayType{Tuple{FT, FT}}(undef, Nx+1, Ny+1)

    arch = grid.architecture
    Oceananigans.Utils.launch!(arch, grid, (Nx+1, Ny+1), _compute_cell_matrix!, cell_matrix, Nx, ℓx, ℓy, grid)

    return cell_matrix
end

flip(::Face) = Center()
flip(::Center) = Face()

@kernel function _compute_cell_matrix!(cell_matrix, Nx, ℓx, ℓy, grid)
    i, j = @index(Global, NTuple)

    vx = flip(ℓx)
    vy = flip(ℓy)

    xl = ξnode(i, j, 1, grid, vx, vy, nothing)
    yl = ηnode(i, j, 1, grid, vx, vy, nothing)

    @inbounds cell_matrix[i, j] = (xl, yl)
end

# Field value functions taken from the regridding benchmarks paper
# https://www.mdpi.com/2297-8747/27/2/31
# Adapted from original Fortran implementations in appendix to Julia.
# May not be exactly correct?  There are slight inconsistencies in the 
# vortex field implementation that I saw.
abstract type ExampleFieldFunction end
Oceananigans.set!(field::Oceananigans.Field, f::ExampleFieldFunction) = set!(field, (lon, lat, z) -> f(lon, lat, z))

struct LongitudeField <: ExampleFieldFunction end
function (::LongitudeField)(lon, lat, z)
    return lon
end

Base.@kwdef struct SinusoidField <: ExampleFieldFunction 
    dp_length::Float64 = 1.2pi
    coef::Float64 = 2
    coefmult::Float64 = 1
end
function (f::SinusoidField)(lon, lat, z)
    return f.coefmult * (f.coef - cos(pi * (acos(cos(deg2rad(lon))) * cos(deg2rad(lat))) / f.dp_length))
end

struct HarmonicField <: ExampleFieldFunction end
function (::HarmonicField)(lon, lat, z)
    return 2 + (sin(2 * deg2rad(lat))^16) * cos(16 * deg2rad(lon))
end

Base.@kwdef struct VortexField <: ExampleFieldFunction
    lon0::Float64 = 5.5
    lat0::Float64 = 0.2
    r0::Float64 = 3.0
    d::Float64 = 5.0
    t::Float64 = 6.0
end
function (f::VortexField)(lon, lat, z)
    # Find the rotated long and lat of the point on (long, lat) on the sphere
    # with the pole at (f.lon0, f.lat0)
    sin_c, cos_c = sincosd(f.lat0)
    sin_lat, cos_lat = sincosd(lat)
    Trm = cos_lat * cos(deg2rad(lon) - f.lon0)
    X = sin_c * Trm - cos_c * sin_lat
    Y = cos_lat * sin(deg2rad(lon) - f.lon0)
    Z = sin_c * sin_lat + cos_c * Trm

    # Recover lat/long
    dlon = atan(Y, X)
    if dlon < 0
        dlon = dlon + 2pi
    end
    dlat = asin(Z)

    Rho = f.r0 * cos(dlat)
    Vt = 3 * sqrt(3)/2/cosh(Rho)/cosh(Rho)*tanh(Rho)
    Omega = Rho == 0 ? 0 : Vt/Rho

    return 2 * (1 + tanh(Rho/f.d * sin(dlon - Omega * f.t)))
end