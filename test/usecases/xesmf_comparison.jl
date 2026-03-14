using ConservativeRegridding
using XESMF
using Oceananigans
using Test
import GeometryOps as GO
using SparseArrays

# ---------------------------------------------------------------------------
# Grid pairs to test
# ---------------------------------------------------------------------------
test_configs = [
    (
        name = "coarse to fine",
        src  = LatitudeLongitudeGrid(size = (18, 9, 1),  longitude = (0, 360), latitude = (-90, 90), z = (0, 1)),
        dst  = LatitudeLongitudeGrid(size = (36, 18, 1), longitude = (0, 360), latitude = (-90, 90), z = (0, 1)),
    ),
    (
        name = "fine to coarse",
        src  = LatitudeLongitudeGrid(size = (36, 18, 1), longitude = (0, 360), latitude = (-90, 90), z = (0, 1)),
        dst  = LatitudeLongitudeGrid(size = (18, 9, 1),  longitude = (0, 360), latitude = (-90, 90), z = (0, 1)),
    ),
    (
        name = "same resolution",
        src  = LatitudeLongitudeGrid(size = (24, 12, 1), longitude = (0, 360), latitude = (-90, 90), z = (0, 1)),
        dst  = LatitudeLongitudeGrid(size = (24, 12, 1), longitude = (0, 360), latitude = (-90, 90), z = (0, 1)),
    ),
]

@testset "XESMF comparison" begin
    for config in test_configs
        @testset "$(config.name)" begin
            src_field = CenterField(config.src)
            dst_field = CenterField(config.dst)

            Nsrc = config.src.Nx * config.src.Ny
            Ndst = config.dst.Nx * config.dst.Ny

            # --- Build both regridders ---
            cr = ConservativeRegridding.Regridder(
                GO.Spherical(), config.dst, config.src; normalize = false,
            )
            xr = XESMF.Regridder(dst_field, src_field; method = "conservative")

            # CR gives raw intersection areas; XESMF gives dst-area-normalised weights.
            # Normalise CR to match: w_cr[d,s] = A[d,s] / dst_areas[d]
            A_cr      = cr.intersections
            dst_areas = cr.dst_areas
            W_xesmf   = xr.weights

            # ---- Cell-by-cell weight comparison ----
            @testset "Weight matrix (cell-by-cell)" begin
                max_abs_diff = 0.0
                max_rel_diff = 0.0
                n_compared   = 0
                n_cr_only    = 0
                n_xesmf_only = 0

                # Compare all nonzero CR entries against XESMF
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

                # Check for entries in XESMF that CR missed
                rows_x = rowvals(W_xesmf)
                for col in 1:size(W_xesmf, 2)
                    for idx in nzrange(W_xesmf, col)
                        row = rows_x[idx]
                        if A_cr[row, col] == 0
                            n_xesmf_only += 1
                        end
                    end
                end

                @test n_compared > 0
                @info "$(config.name) weights" n_compared n_cr_only n_xesmf_only max_abs_diff max_rel_diff
                @test max_rel_diff < 0.05
            end

            # ---- Regridded-field comparison ----
            @testset "Regridded field" begin
                # Smooth test field: f(lon, lat) = cos(lat) * sin(lon)
                set!(src_field, (lon, lat, z) -> cosd(lat) * sind(lon))
                src_data = vec(interior(src_field, :, :, 1))

                # CR regrid
                dst_cr = zeros(Ndst)
                ConservativeRegridding.regrid!(dst_cr, cr, src_data)

                # XESMF regrid
                dst_xesmf = zeros(Ndst)
                xr(dst_xesmf, src_data)

                # Compare where both are nonzero
                covered = (dst_cr .!= 0) .& (dst_xesmf .!= 0)
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
