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
        @test isapprox(sum(weights), expected_area, rtol=1e-5)
    end

    @testset "se_field_to_vec and vec_to_se_field!" begin
        field = Fields.coordinate_field(cubedsphere_space).lat
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

    @testset "inverse_element_map round-trip on GLL nodes (all faces)" begin
        space = CommonSpaces.CubedSphereSpace(;
            radius = GO.Spherical().radius,
            n_quad_points = 4,
            h_elem = 4,
        )
        qs = Spaces.quadrature_style(space)
        ξs, _ = Quadratures.quadrature_points(Float64, qs)
        Nq = length(ξs)
        ne = Topologies.mesh(Spaces.topology(space)).ne

        coords = Fields.coordinate_field(space)
        long_data = parent(Fields.field_values(coords.long))
        lat_data  = parent(Fields.field_values(coords.lat))
        transform = GO.UnitSphereFromGeographic()

        for face in 1:6
            elem_idx = (face - 1) * ne^2 + 1
            for (i, j) in ((1, 1), (Nq, Nq), (2, 3), (3, 2))
                lon = long_data[i, j, 1, elem_idx]
                lat = lat_data[i, j, 1, elem_idx]
                x = transform((lon, lat))
                ξ, η = ClimaCoreExt.inverse_element_map(space, elem_idx, x)
                @test ξ ≈ ξs[i] atol=1e-10
                @test η ≈ ξs[j] atol=1e-10
            end
        end
    end

    @testset "element_jacobian_at returns nodal Jᵉᵢⱼ at GLL nodes" begin
        space = CommonSpaces.CubedSphereSpace(;
            radius = GO.Spherical().radius,
            n_quad_points = 4,
            h_elem = 4,
        )
        qs = Spaces.quadrature_style(space)
        ξs, ws = Quadratures.quadrature_points(Float64, qs)
        Nq = length(ξs)

        WJ = parent(Spaces.weighted_jacobian(space))
        elem_idx = 1
        for j in 1:Nq, i in 1:Nq
            Jᵢⱼ = WJ[i, j, 1, elem_idx] / (ws[i] * ws[j])
            Jq = ClimaCoreExt.element_jacobian_at(space, elem_idx, ξs[i], ξs[j])
            @test Jq ≈ Jᵢⱼ atol=1e-10
        end

        @test ClimaCoreExt.element_jacobian_at(space, elem_idx, 0.0, 0.0) > 0
        @test ClimaCoreExt.element_jacobian_at(space, elem_idx, 0.3, -0.4) > 0
    end

    @testset "accumulate_principled_b: sum equals intersection area" begin
        space = CommonSpaces.CubedSphereSpace(;
            radius = GO.Spherical().radius,
            n_quad_points = 4,
            h_elem = 4,
        )
        Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))

        # When the destination polygon IS the element polygon, k ∩ e = e and
        # Σᵢⱼ B(k, (e,i,j)) = ∫_e Σᵢⱼ ϕᵢϕⱼ dA = ∫_e 1 dA = A_e by partition of unity.
        # This is the principled-version analogue of PDF Eq. 21 collapsed to a
        # single element. Note that individual B[i,j] entries form the FULL mass
        # matrix Mᵢⱼ, not the diagonal Wᵉᵢⱼ — those agree only under GLL quadrature.
        se_tree = Trees.treeify(GO.Spherical(), space)

        for face in 1:6
            ne = Topologies.mesh(Spaces.topology(space)).ne
            elem_idx = (face - 1) * ne^2 + 1
            poly = Trees.getcell(se_tree, elem_idx)
            B = ClimaCoreExt.accumulate_principled_b(
                GO.Spherical(), space, elem_idx, poly;
                triangle_quad_degree = 2 * (Nq - 1),
            )
            @test sum(B) ≈ GO.area(GO.Spherical(), poly) rtol=1e-10
        end
    end

end
