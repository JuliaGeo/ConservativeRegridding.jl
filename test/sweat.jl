# Sweat test for all known tree types against each other

using ConservativeRegridding
using ConservativeRegridding: ConservativeRegridding as CR, Trees
using Test
import GeometryOps as GO, GeoInterface as GI

import ClimaCore, Oceananigans, Healpix, RingGrids

const ClimaCoreExt = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingClimaCoreExt)
const OceananigansExt = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingOceananigansExt)
const HealpixExt = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingHealpixExt)
const RingGridsExt = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingRingGridsExt)

function test_integral_is_conserved(regridder, tree1, values1, tree2, values2, final_values; rtol = sqrt(eps(Float64)))
    tree1_areas = ConservativeRegridding.areas(GO.Spherical(), Trees.treeify(tree1))
    tree2_areas = ConservativeRegridding.areas(GO.Spherical(), Trees.treeify(tree2))

    @test sum(values1 .* tree1_areas) ≈ sum(final_values .* tree2_areas) rtol=rtol
    @test sum(values2 .* tree2_areas) ≈ sum(final_values .* tree1_areas) rtol=rtol
end

function test_intersection_areas_agree(regridder, tree1, tree2; rtol = sqrt(eps(Float64)))
    @test sum(regridder.intersections, dims=2)[:, 1] ≈ regridder.dst_areas rtol=rtol
    @test sum(regridder.intersections, dims=1)[1, :] ≈ regridder.src_areas rtol=rtol
end


function zero_field!(field, values)
    set_field_values!(field, values, (x, y, z = 0) -> 0)
end

function set_field_values!(field::Oceananigans.Field, values, fun)
    Oceananigans.set!(field, fun)
    values .= vec(Oceananigans.interior(field))
end
function set_field_values!(field::ClimaCore.Fields.Field, values, fun)
    space = getfield(field, :space)
    centroids_latlong = GO.UnitSpherical.GeographicFromUnitSphere().(ClimaCoreExt.get_element_centroids(space))
    values .= splat(fun).(centroids_latlong)
    # ClimaCoreExt.set_value_per_element!(field, elems)
end
function set_field_values!(field::Healpix.HealpixMap, values, fun)
    idxs = 1:length(field.pixels)
    vals = (
        begin
            theta, phi = Healpix.pix2ang(field, idx)
            lat = deg2rad(90-theta)
            lon = deg2rad(phi)
            fun(lon, lat)
        end
        for idx in idxs
    )

    values .= vals
end













oceananigans_latlong_grid = Oceananigans.LatitudeLongitudeGrid(size=(360, 180, 1), longitude=(0, 360), latitude=(-90, 90), z = (0, 1), radius = GO.Spherical().radius)
oceananigans_tripolar_grid = Oceananigans.TripolarGrid(size=(360, 180, 1), fold_topology = RightFaceFolded)

oceananigans_latlong_field = Oceananigans.CenterField(oceananigans_latlong_grid)
oceananigans_tripolar_field = Oceananigans.CenterField(oceananigans_tripolar_grid)

oceananigans_latlong_vals = vec(Oceananigans.interior(oceananigans_latlong_field))
oceananigans_tripolar_vals = vec(Oceananigans.interior(oceananigans_tripolar_field))

climacore_cubedsphere_grid = ClimaCore.CommonSpaces.CubedSphereSpace(;
    radius = GO.Spherical().radius,
    n_quad_points = 2,
    h_elem = 64,
)
climacore_cubedsphere_field = ClimaCore.Fields.ones(climacore_cubedsphere_grid)
climacore_cubedsphere_vals = zeros(6*climacore_cubedsphere_grid.grid.topology.mesh.ne^2)

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
climacore_cubedsphere_gilbert_ordered_vals = zeros(6*climacore_cubedsphere_gilbert_ordered_grid.grid.topology.mesh.ne^2)

healpix_nested_order_field = Healpix.HealpixMap{Float64, Healpix.NestedOrder}(64)
healpix_nested_order_vals = healpix_nested_order_field.pixels
healpix_ring_order_field = Healpix.HealpixMap{Float64, Healpix.RingOrder}(64)
healpix_ring_order_vals = healpix_ring_order_field.pixels

oceananigans_fields = [
    ("Oceananigans longitude-latitude grid", oceananigans_latlong_field, oceananigans_latlong_vals),
    ("Oceananigans tripolar grid", oceananigans_tripolar_field, oceananigans_tripolar_vals),
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
for (i, (name1, field1, vals1)) in enumerate(fields)
    for (j, (name2, field2, vals2)) in enumerate(fields)
        @testset "$name1 -> $name2" begin
            tic = time()
            regridder = @test_nowarn ConservativeRegridding.Regridder(GO.Spherical(), field2, field1; normalize = false)
            toc = time()
            push!(regridder_construction_times, (name1, name2) => toc - tic)

            # Test that the areas are correct approximately
            if !(field2 isa Oceananigans.Field && field2.grid isa Oceananigans.TripolarGrid) &&
                !(field1 isa Oceananigans.Field && field1.grid isa Oceananigans.TripolarGrid)
                test_intersection_areas_agree(regridder, field1, field2)
            end

            zero_field!(field1, vals1)
            zero_field!(field2, vals2)

            set_field_values!(field1, vals1, ConservativeRegridding.VortexField(; lat0_rad = deg2rad(80)))
            ConservativeRegridding.regrid!(vals2, regridder, vals1)

            vals2_regridded = vals2[:]
            vals2_analytical = vals2[:]

            # Test that the areas are correct approximately
            if !(field2 isa Oceananigans.Field && field2.grid isa Oceananigans.TripolarGrid) &&
                !(field1 isa Oceananigans.Field && field1.grid isa Oceananigans.TripolarGrid)
                # Oceananigans tripolar grid does not cover the globe
                test_intersection_areas_agree(regridder, field1, field2)
            else
                continue
            end
            i == j && continue

            if field2 isa ClimaCore.Fields.Field # TODO: haven't figured out how this can work yet
                continue
            end

            # if field2 isa Healpix.HealpixMap || field1 isa Healpix.HealpixMap
            #     # Some unknown issue with healpix grids where the regridding to them
            #     # is not conserving the integral?  Not sure what is going on there.
            #     continue
            # end

            @testset "Integral is conserved" begin
                for (fun_name, fun_to_test) in [
                    ("Longitude field", ConservativeRegridding.LongitudeField()),
                    ("Sinusoid field", ConservativeRegridding.SinusoidField()),
                    ("Harmonic field", ConservativeRegridding.HarmonicField()),
                    ("Gulf stream field", ConservativeRegridding.GulfStreamField()),
                    ("Vortex field", ConservativeRegridding.VortexField(; lat0_rad = deg2rad(80))),
                ]
                    @testset "$fun_name" begin
                        set_field_values!(field1, vals1, fun_to_test)
                        # zero_field!(field2, vals2)

                        ConservativeRegridding.regrid!(vals2, regridder, vals1)
                        vals2_regridded .= vals2
                        
                        set_field_values!(field2, vals2, fun_to_test)
                        vals2_analytical .= vals2

                        @test sum(abs.(vals2_regridded) .* regridder.dst_areas) ≈ sum(abs.(vals2_analytical) .* regridder.dst_areas) rtol=1e-2
                    end
                end
            end
        end
    end
end



