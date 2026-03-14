using ConservativeRegridding
using ConservativeRegridding.Trees

import GeoInterface as GI
import GeometryOps as GO
import GeometryOpsCore as GOCore
import GeometryOps: SpatialTreeInterface as STI

using Statistics
using Test

using Oceananigans
using Oceananigans.Operators: Δxᶜᶠᵃ, Δyᶠᶜᵃ, Δzᶜᶠᶜ, Δzᶠᶜᶜ, extrinsic_vector
using Oceananigans.Grids: RightFaceFolded

@inline geometry_grid(grid::ImmersedBoundaryGrid) = grid.underlying_grid
@inline geometry_grid(grid) = grid

function center_horizontal_size(grid)
    g = geometry_grid(grid)
    Nx, Ny, _ = size(g)
    if g isa Oceananigans.OrthogonalSphericalShellGrids.TripolarGrid{<:Any, <:Any, RightFaceFolded}
        return Nx, Ny - 1
    else
        return Nx, Ny
    end
end

@inline function basis_u(grid, i, j)
    g = geometry_grid(grid)
    g isa LatitudeLongitudeGrid && return 1.0, 0.0
    Nx, Ny = center_horizontal_size(grid)
    ii = clamp(i, 1, Nx)
    jj = clamp(j, 1, Ny)
    return extrinsic_vector(ii, jj, 1, g, 1.0, 0.0)
end

@inline function basis_v(grid, i, j)
    g = geometry_grid(grid)
    g isa LatitudeLongitudeGrid && return 0.0, 1.0
    Nx, Ny = center_horizontal_size(grid)
    ii = clamp(i, 1, Nx)
    jj = clamp(j, 1, Ny)
    return extrinsic_vector(ii, jj, 1, g, 0.0, 1.0)
end

"""
Return the global eastward and northward integrated transport represented by a pair
of C-grid face velocity fields `(u, v)`.

This is the conserved vector quantity for the line-integral + rotation remap:
source intrinsic `(u, v)` are first converted to face transports and then projected
onto extrinsic east/north directions before global summation.
"""
function total_east_north_transport(u_field, v_field)
    @assert u_field.grid === v_field.grid
    grid = u_field.grid
    u_data = interior(u_field)
    v_data = interior(v_field)

    Nxᵤ, Nyᵤ, Nz = size(u_field)
    Nxᵥ, Nyᵥ, _ = size(v_field)

    east_total = 0.0
    north_total = 0.0

    for k in 1:Nz, j in 1:Nyᵤ, i in 1:Nxᵤ
        F = u_data[i, j, k] * Δyᶠᶜᵃ(i, j, k, grid) * Δzᶠᶜᶜ(i, j, k, grid)
        ex, ny = basis_u(grid, i, j)
        east_total += F * ex
        north_total += F * ny
    end

    for k in 1:Nz, j in 1:Nyᵥ, i in 1:Nxᵥ
        F = v_data[i, j, k] * Δxᶜᶠᵃ(i, j, k, grid) * Δzᶜᶠᶜ(i, j, k, grid)
        ex, ny = basis_v(grid, i, j)
        east_total += F * ex
        north_total += F * ny
    end

    return east_total, north_total
end


@testset "Lat-long upscaling" begin
    coarse_grid = LatitudeLongitudeGrid(size=(90, 45, 1),   longitude=(0, 360), latitude=(-90, 90), z=(0, 1))
    fine_grid   = LatitudeLongitudeGrid(size=(360, 180, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))

    dst = CenterField(coarse_grid)
    src = CenterField(fine_grid)

    set!(src, (x, y, z) -> x)

    regridder = ConservativeRegridding.Regridder(dst, src)

    ConservativeRegridding.regrid!(vec(interior(dst)), regridder, vec(interior(src)))

    @test mean(interior(dst)) == mean(interior(src))

    set!(dst, (x, y, z) -> rand())

    ConservativeRegridding.regrid!(vec(interior(src)), transpose(regridder), vec(interior(dst)))

    @test mean(dst) ≈ mean(src) rtol=1e-5
end

@testset "Tripolar (FPivot / RightFaceFolded) to lat-long" begin
    dst_grid = LatitudeLongitudeGrid(size=(360, 180, 1),   longitude=(0, 360), latitude=(-90, 90), z=(0, 1))
    src_grid = TripolarGrid(size=(360, 180, 1), fold_topology = RightFaceFolded)

    dst = CenterField(dst_grid)
    src = CenterField(src_grid)

    regridder = ConservativeRegridding.Regridder(dst, src)

    set!(src, (x, y, z) -> abs(x))
    set!(dst, (x, y, z) -> 0)

    ConservativeRegridding.regrid!(vec(interior(dst)), regridder, vec(interior(src)))

    @test sum(vec(interior(dst)) .* regridder.dst_areas) ≈ sum(vec(interior(src)) .* regridder.src_areas) rtol=1e-10

    set!(src, (x, y, z) -> rand())
    ConservativeRegridding.regrid!(vec(interior(dst)), regridder, vec(interior(src)))
    @test sum(vec(interior(dst)) .* regridder.dst_areas) ≈ sum(vec(interior(src)) .* regridder.src_areas) rtol=1e-7
end

@testset "Tripolar (UPivot / RightCenterFolded) to lat-long" begin
    dst_grid = LatitudeLongitudeGrid(size=(360, 180, 1),   longitude=(0, 360), latitude=(-90, 90), z=(0, 1))
    src_grid = TripolarGrid(size=(360, 180, 1), fold_topology = RightCenterFolded)

    dst = CenterField(dst_grid)
    src = CenterField(src_grid)

    regridder = ConservativeRegridding.Regridder(dst, src)

    set!(src, (x, y, z) -> abs(x))
    set!(dst, (x, y, z) -> 0)

    ConservativeRegridding.regrid!(vec(interior(dst)), regridder, vec(interior(src)))

    # The tests on a `RightCenterFolded` grid have a lower relative tolerance because of the extra 
    # half row thatis repeated at the top of the domain
    @test sum(vec(interior(dst)) .* regridder.dst_areas) ≈ sum(vec(interior(src)) .* regridder.src_areas) rtol=1e-7

    set!(src, (x, y, z) -> rand())
    ConservativeRegridding.regrid!(vec(interior(dst)), regridder, vec(interior(src)))
    @test sum(vec(interior(dst)) .* regridder.dst_areas) ≈ sum(vec(interior(src)) .* regridder.src_areas) rtol=1e-7
end

@testset "Face-located regridding and transport closure (Tripolar to lat-long)" begin
    dst_grid = LatitudeLongitudeGrid(size=(128, 64, 1),
                                     longitude=(0, 360),
                                     latitude=(-90, 90),
                                     z=(0, 1))

    tripolar_size(::Type{RightFaceFolded}) = (96, 65, 1)
    tripolar_size(::Type{RightCenterFolded}) = (96, 64, 1)

    for fold_topology in (RightFaceFolded, RightCenterFolded)
        @testset "$(fold_topology) source topology" begin
            src_grid = TripolarGrid(size=tripolar_size(fold_topology), fold_topology=fold_topology)

            src_u = XFaceField(src_grid)
            src_v = YFaceField(src_grid)
            dst_u = XFaceField(dst_grid)
            dst_v = YFaceField(dst_grid)

            set!(src_u, (λ, φ, z) -> isfinite(λ) && isfinite(φ) ? cosd(φ) * sind(2λ) : 0.0)
            set!(src_v, (λ, φ, z) -> isfinite(λ) && isfinite(φ) ? sind(φ) * cosd(3λ) : 0.0)

            Rvel = ConservativeRegridding.VelocityLineIntegralRegridder(dst_u, dst_v, src_u, src_v)
            ConservativeRegridding.regrid_velocity_transport!(dst_u, dst_v, Rvel, src_u, src_v)

            src_east, src_north = total_east_north_transport(src_u, src_v)
            dst_east, dst_north = total_east_north_transport(dst_u, dst_v)

            @test dst_east ≈ src_east rtol=1e-6 atol=1e-12
            @test dst_north ≈ src_north rtol=1e-6 atol=1e-12
        end
    end
end

@testset "Immersed-boundary wet-fraction closure (Tripolar to lat-long)" begin
    dst_underlying = LatitudeLongitudeGrid(size=(128, 64, 1),
                                           longitude=(0, 360),
                                           latitude=(-90, 90),
                                           z=(-1, 0))

    bottom_height(ξ, η) = isfinite(ξ) && isfinite(η) ? clamp(-0.82 + 0.42 * cosd(η) * sind(ξ), -1.0, -0.02) : -0.02
    tripolar_size(::Type{RightFaceFolded}) = (96, 65, 1)
    tripolar_size(::Type{RightCenterFolded}) = (96, 64, 1)

    for fold_topology in (RightFaceFolded, RightCenterFolded)
        @testset "$(fold_topology) source topology" begin
            src_underlying = TripolarGrid(size=tripolar_size(fold_topology),
                                          fold_topology=fold_topology,
                                          z=(-1, 0))

            src_grid = ImmersedBoundaryGrid(src_underlying, PartialCellBottom(bottom_height; minimum_fractional_cell_height=0.05))
            dst_grid = ImmersedBoundaryGrid(dst_underlying, PartialCellBottom(bottom_height; minimum_fractional_cell_height=0.05))

            src_tracer = CenterField(src_grid)
            dst_tracer = CenterField(dst_grid)

            src_u = XFaceField(src_grid)
            src_v = YFaceField(src_grid)
            dst_u = XFaceField(dst_grid)
            dst_v = YFaceField(dst_grid)

            set!(src_tracer, (λ, φ, z) -> isfinite(λ) && isfinite(φ) ? 1 + 0.2cosd(φ) * cosd(2λ) : 0.0)
            set!(src_u, (λ, φ, z) -> isfinite(λ) && isfinite(φ) ? cosd(φ) * sind(2λ) : 0.0)
            set!(src_v, (λ, φ, z) -> isfinite(λ) && isfinite(φ) ? sind(φ) * cosd(3λ) : 0.0)

            tracer_regridder = ConservativeRegridding.Regridder(dst_tracer, src_tracer)
            Rvel = ConservativeRegridding.VelocityLineIntegralRegridder(dst_u, dst_v, src_u, src_v)

            ConservativeRegridding.regrid!(vec(interior(dst_tracer)), tracer_regridder, vec(interior(src_tracer)))
            ConservativeRegridding.regrid_velocity_transport!(dst_u, dst_v, Rvel, src_u, src_v)

            src_tracer_mass = sum(vec(interior(src_tracer)) .* tracer_regridder.src_areas)
            dst_tracer_mass = sum(vec(interior(dst_tracer)) .* tracer_regridder.dst_areas)

            src_east, src_north = total_east_north_transport(src_u, src_v)
            dst_east, dst_north = total_east_north_transport(dst_u, dst_v)

            @test dst_tracer_mass ≈ src_tracer_mass rtol=1e-6 atol=1e-12
            @test dst_east ≈ src_east rtol=1e-6 atol=1e-12
            @test dst_north ≈ src_north rtol=1e-6 atol=1e-12
        end
    end
end

@testset "Rectilinear (planar) upscaling" begin
    large_domain_grid = RectilinearGrid(size=(100, 100), x=(0, 2), y=(0, 2), topology=(Periodic, Periodic, Flat))
    small_domain_grid = RectilinearGrid(size=(200, 200), x=(0, 1), y=(0, 1), topology=(Periodic, Periodic, Flat))

    src = CenterField(small_domain_grid)
    dst = CenterField(large_domain_grid)

    set!(src, 1)

    regridder = ConservativeRegridding.Regridder(dst, src; threaded = false)

    ConservativeRegridding.regrid!(vec(interior(dst)), regridder, vec(interior(src)))

    # Compute the integral and make sure it is the same as the original field
    dst_int = Field(Integral(dst))
    src_int = Field(Integral(src))

    compute!(dst_int)
    compute!(src_int)

    @test only(dst_int) ≈ only(src_int)
end
