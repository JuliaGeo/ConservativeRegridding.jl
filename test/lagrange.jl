using ClimaCore: Quadratures
using StaticArrays
using Test

# Nodal Lagrange evaluation matches `ConservativeRegriddingClimaCoreExt` (barycentric formula).

@testset "Lagrange basis via ClimaCore.Quadratures.interpolation_matrix" begin
    ξs, _ = CCQ.quadrature_points(Float64, Quadratures.GLL{4}())

    @testset "Kronecker delta at nodes" begin
        for p in 1:4
            M = Quadratures.interpolation_matrix(SVector(ξs[p]), ξs)
            for i in 1:4
                @test M[1, i] ≈ (i == p ? 1.0 : 0.0) atol=1e-12
            end
        end
    end

    @testset "Partition of unity off-node" begin
        for ξ in (-0.7, -0.3, 0.0, 0.42, 0.91)
            M = CCQ.interpolation_matrix(SVector(ξ), ξs)
            @test sum(M[1, :]) ≈ 1.0 atol=1e-12
        end
    end

    @testset "Single-point row is length Nq" begin
        ξ = 0.3
        M = Quadratures.interpolation_matrix(SVector(ξ), ξs)
        @test size(M) == (1, 4)
        @test sum(M[1, :]) ≈ 1.0 atol=1e-12
    end

    @testset "Buffer fill from matrix row" begin
        ξs3, _ = Quadratures.quadrature_points(Float64, CCQ.GLL{3}())
        ξ = 0.5
        M = CCQ.interpolation_matrix(SVector(ξ), ξs3)
        out = zeros(3)
        out .= M[1, :]
        @test out ≈ Vector(M[1, :])
    end
end
