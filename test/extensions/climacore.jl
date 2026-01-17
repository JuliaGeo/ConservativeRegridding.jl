using ConservativeRegridding
using ConservativeRegridding: Trees
using Statistics
using Test
import GeometryOps as GO, GeoInterface as GI, LibGEOS

using ClimaCore:
    CommonSpaces, Fields, Spaces, RecursiveApply, Meshes, Quadratures, Topologies, Domains, ClimaComms

const ClimaCoreExt = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingClimaCoreExt)


@testset "Test helper functions in extension" begin
    cubedsphere_space = CommonSpaces.CubedSphereSpace(;
        radius = latlon_grid.radius,
        n_quad_points = 2,
        h_elem = 64,
    )

    # Define a field on the first space, to use as our source field
    field = Fields.coordinate_field(cubedsphere_space).long
    ones_field = Fields.ones(cubedsphere_space)
    cubed_sphere_vals = zeros(6*cubedsphere_space.grid.topology.mesh.ne^2)
    ClimaCoreExt.get_value_per_element!(cubed_sphere_vals, field, ones_field)

    @testset "integrate_each_element" begin
        @test_broken isapprox(
            sum(ClimaCoreExt.integrate_each_element(field)), 
            sum(field), 
            atol = 1e-11
        )
        @test_broken sum(ClimaCoreExt.integrate_each_element(ones_field)) ≈ sum(ones_field)
    end
    @testset "Convert to one value per element" begin
        value_per_element1 = zeros(Float64, Meshes.nelements(cubedsphere_space.grid.topology.mesh))
        field1_one_value_per_element = Fields.zeros(cubedsphere_space)
        ClimaCoreExt.set_value_per_element!(field1_one_value_per_element, value_per_element1)

        # Check the error of converting to one value per element
        abs_error_one_value_per_element = abs(sum(field1_one_value_per_element) - sum(field))
        @test_broken abs_error_one_value_per_element < 2e-11
        @test_broken isapprox(mean(field), mean(field1_one_value_per_element), atol=1e-14)
    end
end