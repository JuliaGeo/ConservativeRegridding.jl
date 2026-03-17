using ConservativeRegridding
using XESMF
using Oceananigans
using Oceananigans.Grids: RightCenterFolded
using Test
import GeometryOps as GO
using SparseArrays

const R = 1.0  # unit sphere radius for all grids

# ---------------------------------------------------------------------------
# LatLon ↔ LatLon grid pairs (cell ordering matches exactly)
# ---------------------------------------------------------------------------
latlon_configs = [
    (
        name = "coarse to fine",
        src  = LatitudeLongitudeGrid(size = (18, 9, 1),  longitude = (0, 360), latitude = (-90, 90), z = (0, 1), radius = R),
        dst  = LatitudeLongitudeGrid(size = (36, 18, 1), longitude = (0, 360), latitude = (-90, 90), z = (0, 1), radius = R),
    ),
    (
        name = "fine to coarse",
        src  = LatitudeLongitudeGrid(size = (36, 18, 1), longitude = (0, 360), latitude = (-90, 90), z = (0, 1), radius = R),
        dst  = LatitudeLongitudeGrid(size = (18, 9, 1),  longitude = (0, 360), latitude = (-90, 90), z = (0, 1), radius = R),
    ),
    (
        name = "same resolution",
        src  = LatitudeLongitudeGrid(size = (24, 12, 1), longitude = (0, 360), latitude = (-90, 90), z = (0, 1), radius = R),
        dst  = LatitudeLongitudeGrid(size = (24, 12, 1), longitude = (0, 360), latitude = (-90, 90), z = (0, 1), radius = R),
    ),
]

# ---------------------------------------------------------------------------
# Tripolar ↔ LatLon pairs
#
# Cell ordering for RightCenterFolded tripolar differs between CR and XESMF,
# so we compare using constant-field regridding (ordering-invariant).
#
# Additionally, XESMF does not know about the fold topology, so it
# double-counts fold-row cells.  We exclude those from the comparison.
# ---------------------------------------------------------------------------
tripolar_configs = [
    (
        name = "tripolar to latlon",
        src  = TripolarGrid(size = (40, 20, 1), fold_topology = RightCenterFolded, radius = R),
        dst  = LatitudeLongitudeGrid(size = (36, 18, 1), longitude = (0, 360), latitude = (-90, 90), z = (0, 1), radius = R),
    ),
    (
        name = "latlon to tripolar",
        src  = LatitudeLongitudeGrid(size = (36, 18, 1), longitude = (0, 360), latitude = (-90, 90), z = (0, 1), radius = R),
        dst  = TripolarGrid(size = (40, 20, 1), fold_topology = RightCenterFolded, radius = R),
    ),
]

@testset "XESMF comparison" begin
    # =======================================================================
    # LatLon ↔ LatLon: full cell-by-cell comparison
    # =======================================================================
    @testset "LatLon ↔ LatLon" begin
        for config in latlon_configs
            @testset "$(config.name)" begin
                src_field = CenterField(config.src)
                dst_field = CenterField(config.dst)
                Nsrc = config.src.Nx * config.src.Ny
                Ndst = config.dst.Nx * config.dst.Ny

                # Manifold is inferred from grid radius via best_manifold
                cr = ConservativeRegridding.Regridder(
                    config.dst, config.src; normalize = false,
                )
                xr = XESMF.Regridder(dst_field, src_field; method = "conservative")

                A_cr      = cr.intersections
                dst_areas = cr.dst_areas
                W_xesmf   = xr.weights

                @testset "Weight matrix (cell-by-cell)" begin
                    max_abs_diff = 0.0
                    max_rel_diff = 0.0
                    n_compared   = 0
                    n_cr_only    = 0
                    n_xesmf_only = 0

                    for d in 1:Ndst, s in 1:Nsrc
                        a = A_cr[d, s]
                        a == 0 && continue
                        w_cr = a / dst_areas[d]
                        w_x  = W_xesmf[d, s]
                        if w_x == 0
                            n_cr_only += 1
                            continue
                        end
                        abs_diff = abs(w_cr - w_x)
                        denom    = max(abs(w_x), abs(w_cr), 1e-15)
                        max_abs_diff = max(max_abs_diff, abs_diff)
                        max_rel_diff = max(max_rel_diff, abs_diff / denom)
                        n_compared += 1
                    end

                    rows_x = rowvals(W_xesmf)
                    for col in 1:size(W_xesmf, 2)
                        for idx in nzrange(W_xesmf, col)
                            if A_cr[rows_x[idx], col] == 0
                                n_xesmf_only += 1
                            end
                        end
                    end

                    @test n_compared > 0
                    @info "$(config.name) weights" n_compared n_cr_only n_xesmf_only max_abs_diff max_rel_diff
                    @test max_rel_diff < 0.05
                end

                @testset "Regridded field" begin
                    set!(src_field, (lon, lat, z) -> cosd(lat) * sind(lon))
                    src_data = vec(interior(src_field, :, :, 1))

                    dst_cr    = zeros(Ndst)
                    dst_xesmf = zeros(Ndst)
                    ConservativeRegridding.regrid!(dst_cr, cr, src_data)
                    xr(dst_xesmf, src_data)

                    covered   = (dst_cr .!= 0) .& (dst_xesmf .!= 0)
                    @test any(covered)
                    if any(covered)
                        abs_diffs = abs.(dst_cr[covered] .- dst_xesmf[covered])
                        denom     = max.(abs.(dst_cr[covered]), abs.(dst_xesmf[covered]), 1e-15)
                        rel_diffs = abs_diffs ./ denom
                        max_rel   = maximum(rel_diffs)
                        mean_rel  = sum(rel_diffs) / length(rel_diffs)
                        @info "$(config.name) regrid" max_rel mean_rel
                        @test max_rel < 0.05
                    end
                end
            end
        end
    end

    # =======================================================================
    # Tripolar ↔ LatLon: constant-field comparison (ordering-invariant)
    #
    # For tripolar grids the cell ordering in CR's sparse matrix differs from
    # XESMF's column-major ordering, so we regrid a constant field (ones),
    # which is invariant to reordering.  Fold-affected destination cells
    # (where XESMF double-counts) are excluded from the comparison.
    # =======================================================================
    @testset "Tripolar ↔ LatLon" begin
        for config in tripolar_configs
            @testset "$(config.name)" begin
                src_field = CenterField(config.src)
                dst_field = CenterField(config.dst)
                Nsrc = config.src.Nx * config.src.Ny
                Ndst = config.dst.Nx * config.dst.Ny

                cr = ConservativeRegridding.Regridder(
                    config.dst, config.src; normalize = false,
                )
                xr = XESMF.Regridder(dst_field, src_field; method = "conservative")

                @testset "Constant field" begin
                    dst_cr    = zeros(Ndst)
                    dst_xesmf = zeros(Ndst)
                    ConservativeRegridding.regrid!(dst_cr, cr, ones(Nsrc))
                    xr(dst_xesmf, ones(Nsrc))

                    # Exclude fold-affected cells (XESMF > 1.001 indicates partial
                    # fold contamination) and ghost cells (CR NaN/0)
                    clean = (dst_xesmf .> 0.01) .& (dst_xesmf .< 1.001) .&
                            isfinite.(dst_cr) .& (dst_cr .> 0.01)
                    n_clean = count(clean)
                    @test n_clean > 0

                    if n_clean > 0
                        diffs    = abs.(dst_cr[clean] .- dst_xesmf[clean])
                        max_diff = maximum(diffs)
                        mean_diff = sum(diffs) / n_clean

                        n_fold_affected = count(dst_xesmf .> 1.001)
                        n_ghost_nan    = count(.!isfinite.(dst_cr))

                        @info "$(config.name) constant field" n_clean n_fold_affected n_ghost_nan max_diff mean_diff
                        @test max_diff < 1e-13
                    end
                end
            end
        end
    end
end
