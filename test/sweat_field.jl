# Field-path sweat test.
#
# Mirrors `sweat.jl` but exercises `regrid!(field2, regridder, field1)` — the
# full pipeline that dispatches on the field type for `extract_*_arraylike` /
# `initialize_regridding!` / `finalize_regridding!`. The vec-path sweat shows
# the regridder matrix is correct; this one shows the package-native plumbing
# (Oceananigans `interior`, ClimaCore `se_field_to_vec`/`vec_to_se_field!`,
# Healpix `parent(map)`) works end-to-end. The fold
# mirror in `OceananigansExt.finalize_regridding!` only runs on this path, so
# regridding INTO a `RightCenterFolded` tripolar grid should pass here even
# though it NaNs in the vec sweat.

using ConservativeRegridding
using ConservativeRegridding: ConservativeRegridding as CR, Trees
using Test
import GeometryOps as GO, GeoInterface as GI

import ClimaCore, Oceananigans, Healpix, RingGrids

const ClimaCoreExt = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingClimaCoreExt)
const OceananigansExt = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingOceananigansExt)
const HealpixExt = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingHealpixExt)
const RingGridsExt = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingRingGridsExt)

include("sweat_common.jl")

# ---- push_to_field! / pull_from_field! ----
#
# Bridge between the per-cell `vals` vector (which `set_field_values!` writes
# into via quadrature) and the field's native storage. Each is conceptually
# the inverse of how the package's `initialize_regridding!` / `finalize_regridding!`
# routes data through `regridder.src_temp`/`dst_temp` for that field type.

# Oceananigans: `interior(field)` is the halo-stripped 3D view; `vec` linearizes
# it in the same order as `vals`.
push_to_field!(field::Oceananigans.AbstractField, vals) = (copyto!(vec(Oceananigans.interior(field)), vals); field)
pull_from_field!(vals, field::Oceananigans.AbstractField) = (copyto!(vals, vec(Oceananigans.interior(field))); vals)

push_to_field!(field::ClimaCore.Fields.Field, vals) = (ClimaCoreExt.vec_to_se_field!(field, vals); field)
pull_from_field!(vals, field::ClimaCore.Fields.Field) = (vals .= ClimaCoreExt.se_field_to_vec(field); vals)

# Healpix: the field IS a wrapped vector — `parent` exposes the same storage
# the Healpix extension forwards to in the pipeline.
push_to_field!(field::Healpix.HealpixMap, vals) = (copyto!(parent(field), vals); field)
pull_from_field!(vals, field::Healpix.HealpixMap) = (copyto!(vals, parent(field)); vals)

# ---- Grid / field / vals setup (mirrors sweat.jl) ----

oceananigans_latlong_grid = Oceananigans.LatitudeLongitudeGrid(size=(360, 180, 1), longitude=(0, 360), latitude=(-90, 90), z = (0, 1), radius = GO.Spherical().radius)
oceananigans_tripolar_grid = Oceananigans.TripolarGrid(size=(360, 180, 1), fold_topology = Oceananigans.RightFaceFolded)
oceananigans_tripolarC_grid = Oceananigans.TripolarGrid(size=(360, 180, 1), fold_topology = Oceananigans.RightCenterFolded)
oceananigans_rotated_latlong_grid = Oceananigans.RotatedLatitudeLongitudeGrid(size=(90, 40, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1), north_pole=(70, 55))

oceananigans_latlong_field = Oceananigans.CenterField(oceananigans_latlong_grid)
oceananigans_tripolar_field = Oceananigans.CenterField(oceananigans_tripolar_grid)
oceananigans_tripolarC_field = Oceananigans.CenterField(oceananigans_tripolarC_grid)
oceananigans_rotated_latlong_field = Oceananigans.CenterField(oceananigans_rotated_latlong_grid)

oceananigans_latlong_vals = vec(Oceananigans.interior(oceananigans_latlong_field))
oceananigans_tripolar_vals = vec(Oceananigans.interior(oceananigans_tripolar_field))
oceananigans_tripolarC_vals = vec(Oceananigans.interior(oceananigans_tripolarC_field))
oceananigans_rotated_latlong_vals = vec(Oceananigans.interior(oceananigans_rotated_latlong_field))

climacore_cubedsphere_grid = ClimaCore.CommonSpaces.CubedSphereSpace(;
    radius = GO.Spherical().radius,
    n_quad_points = 2,
    h_elem = 64,
)
climacore_cubedsphere_field = ClimaCore.Fields.ones(climacore_cubedsphere_grid)
climacore_cubedsphere_vals = zeros(length(ClimaCoreExt.se_field_to_vec(climacore_cubedsphere_field)))

climacore_cubedsphere_gilbert_ordered_grid = let
    device = ClimaCore.ClimaComms.device()
    context = ClimaCore.ClimaComms.context(device)
    h_elem = 64
    h_mesh = ClimaCore.Meshes.EquiangularCubedSphere(ClimaCore.Domains.SphereDomain{Float64}(GO.Spherical().radius), h_elem)
    h_topology = ClimaCore.Topologies.Topology2D(context, h_mesh, ClimaCore.Topologies.spacefillingcurve(h_mesh))
    ClimaCore.CommonSpaces.CubedSphereSpace(;
        radius = h_mesh.domain.radius,
        n_quad_points = 2,
        h_elem = h_elem,
        h_mesh = h_mesh,
        h_topology = h_topology,
    )
end
climacore_cubedsphere_gilbert_ordered_field = ClimaCore.Fields.ones(climacore_cubedsphere_gilbert_ordered_grid)
climacore_cubedsphere_gilbert_ordered_vals = zeros(length(ClimaCoreExt.se_field_to_vec(climacore_cubedsphere_gilbert_ordered_field)))

healpix_nested_order_field = Healpix.HealpixMap{Float64, Healpix.NestedOrder}(64)
healpix_nested_order_vals = healpix_nested_order_field.pixels
healpix_ring_order_field = Healpix.HealpixMap{Float64, Healpix.RingOrder}(64)
healpix_ring_order_vals = healpix_ring_order_field.pixels

oceananigans_fields = [
    ("Oceananigans longitude-latitude grid", oceananigans_latlong_field, oceananigans_latlong_vals),
    ("Oceananigans tripolar grid (RightFaceFolded)", oceananigans_tripolar_field, oceananigans_tripolar_vals),
    ("Oceananigans tripolar grid (RightCenterFolded)", oceananigans_tripolarC_field, oceananigans_tripolarC_vals),
    ("Oceananigans rotated longitude-latitude grid", oceananigans_rotated_latlong_field, oceananigans_rotated_latlong_vals),
]

healpix_fields = [
    ("Healpix nested order grid", healpix_nested_order_field, healpix_nested_order_vals),
    ("Healpix ring order grid", healpix_ring_order_field, healpix_ring_order_vals),
]

climacore_fields = [
    ("ClimaCore cubed sphere grid", climacore_cubedsphere_field, climacore_cubedsphere_vals),
    ("ClimaCore cubed sphere grid (Gilbert ordered)", climacore_cubedsphere_gilbert_ordered_field, climacore_cubedsphere_gilbert_ordered_vals),
]

fields = [oceananigans_fields..., climacore_fields..., healpix_fields...]

regridder_construction_times = Pair{Tuple{String, String}, Float64}[]
@testset "Sweat test (field path)" begin
    @testset "Sweat test (field path): $name1 -> $name2" for (i, (name1, field1, vals1)) in enumerate(fields), (j, (name2, field2, vals2)) in enumerate(fields)
        # SE → SE is unsupported
        both_se = field1 isa ClimaCore.Fields.Field && field2 isa ClimaCore.Fields.Field
        both_se && continue

        tic = time()
        regridder = @test_nowarn ConservativeRegridding.Regridder(GO.Spherical(), field2, field1; normalize = false)
        toc = time()
        push!(regridder_construction_times, (name1, name2) => toc - tic)

        # The intersection-area agreement check is an FV-overlap property; the
        # principled/L2 SE matrices are not pure overlap matrices, so skip it
        # when an SE field is involved (SE conservation is checked below).
        has_se = field1 isa ClimaCore.Fields.Field || field2 isa ClimaCore.Fields.Field
        has_tripolar = (field2 isa Oceananigans.Field && field2.grid isa Oceananigans.TripolarGrid) ||
                       (field1 isa Oceananigans.Field && field1.grid isa Oceananigans.TripolarGrid)
        has_rotated = (field2 isa Oceananigans.Field && field2.grid isa Oceananigans.RotatedLatitudeLongitudeGrid) ||
                      (field1 isa Oceananigans.Field && field1.grid isa Oceananigans.RotatedLatitudeLongitudeGrid)
        areas_rtol = has_rotated ? 1e-2 : sqrt(eps(Float64))
        if !has_tripolar && !has_se
            test_intersection_areas_agree(regridder, field1, field2; rtol=areas_rtol)
        end

        zero_field!(field1, vals1); push_to_field!(field1, vals1)
        zero_field!(field2, vals2); push_to_field!(field2, vals2)

        set_field_values!(field1, vals1, ConservativeRegridding.VortexField(; lat0_rad = deg2rad(80)))
        push_to_field!(field1, vals1)
        ConservativeRegridding.regrid!(field2, regridder, field1)
        pull_from_field!(vals2, field2)

        vals2_regridded = vals2[:]
        vals2_analytical = vals2[:]

        if !has_tripolar && !has_se
            test_intersection_areas_agree(regridder, field1, field2; rtol=areas_rtol)
        end
        i == j && continue

        @testset "Integral is conserved w.r.t. analytical values" begin
            for (fun_name, fun_to_test) in [
                ("Longitude field", ConservativeRegridding.LongitudeField()),
                ("Sinusoid field", ConservativeRegridding.SinusoidField()),
                ("Harmonic field", ConservativeRegridding.HarmonicField()),
                ("Gulf stream field", ConservativeRegridding.GulfStreamField()),
                ("Vortex field", ConservativeRegridding.VortexField(; lat0_rad = deg2rad(80))),
            ]
                @testset "$fun_name" begin
                    set_field_values!(field1, vals1, fun_to_test)
                    push_to_field!(field1, vals1)

                    ConservativeRegridding.regrid!(field2, regridder, field1)
                    pull_from_field!(vals2, field2)
                    vals2_regridded .= vals2

                    set_field_values!(field2, vals2, fun_to_test)
                    vals2_analytical .= vals2

                    straddles_dateline(f) =
                        (f isa Oceananigans.Field && (f.grid isa Oceananigans.TripolarGrid ||
                                                      f.grid isa Oceananigans.RotatedLatitudeLongitudeGrid)) ||
                        f isa ClimaCore.Fields.Field
                    has_dateline_seam = straddles_dateline(field1) || straddles_dateline(field2)
                    tol = (has_dateline_seam && fun_to_test isa ConservativeRegridding.LongitudeField) ? 5e-2 : 1e-2
                    w2 = test_integration_weights(field2, regridder)
                    # A non-global source (tripolar/rotated) leaves destination cells outside its footprint with no contribution, 
                    # so the regridder zeros them while the analytical field is defined everywhere. Compare the conserved integral
                    # only over the covered destination cells (rows with nonzero intersection).
                    covered = vec(sum(regridder.intersections; dims=2)) .> 0
                    @test sum(abs.(vals2_regridded[covered]) .* w2[covered]) ≈ sum(abs.(vals2_analytical[covered]) .* w2[covered]) rtol=tol
                end
            end
        end
    end
end
