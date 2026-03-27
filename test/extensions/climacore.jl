using ConservativeRegridding
using ConservativeRegridding: Trees
using Statistics
using Test
import GeometryOps as GO, GeoInterface as GI, LibGEOS

using ClimaCore:
    CommonSpaces, Fields, Spaces, RecursiveApply, Meshes, Quadratures, Topologies, Domains, ClimaComms

const ClimaCoreExt = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingClimaCoreExt)


@testset "Test SE node helpers" begin
    cubedsphere_space = CommonSpaces.CubedSphereSpace(;
        radius = GO.Spherical().radius,
        n_quad_points = 3,
        h_elem = 16,
    )

    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(cubedsphere_space))
    Nh = Meshes.nelements(Topologies.mesh(Spaces.topology(cubedsphere_space)))
    N_nodes = Nq^2 * Nh

    @testset "se_node_positions" begin
        positions = ClimaCoreExt.se_node_positions(cubedsphere_space)
        @test length(positions) == N_nodes
        # All points should be on the unit sphere
        @test all(p -> isapprox(sum(x -> x^2, p), 1.0; atol=1e-12), positions)
    end

    @testset "se_node_weights" begin
        weights = ClimaCoreExt.se_node_weights(cubedsphere_space)
        @test length(weights) == N_nodes
        @test all(w -> w > 0, weights)
        # Sum of weights should equal the surface area of the sphere (4πr²)
        radius = GO.Spherical().radius
        expected_area = 4π * radius^2
        @test isapprox(sum(weights), expected_area, rtol=1e-10)
    end

    @testset "se_field_to_vec and vec_to_se_field!" begin
        field = Fields.coordinate_field(cubedsphere_space).long
        v = ClimaCoreExt.se_field_to_vec(field)
        @test length(v) == N_nodes

        roundtrip = Fields.zeros(cubedsphere_space)
        ClimaCoreExt.vec_to_se_field!(roundtrip, v)
        roundtrip_v = ClimaCoreExt.se_field_to_vec(roundtrip)
        @test roundtrip_v ≈ v

        # Weighted integral via sum(field) should match manual dot(weights, values)
        weights = ClimaCoreExt.se_node_weights(cubedsphere_space)
        @test isapprox(sum(weights .* v), sum(field), rtol=1e-10)
    end
end
