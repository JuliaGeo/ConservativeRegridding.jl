using ConservativeRegridding: Lagrange
using Test

@testset "Lagrange basis on GLL nodes" begin
    # GLL(4) nodes: known exact values
    ξs = [-1.0, -sqrt(1/5), sqrt(1/5), 1.0]

    @testset "Kronecker delta at nodes" begin
        for i in 1:4, p in 1:4
            ϕᵢ_at_ξₚ = Lagrange.evaluate(ξs, i, ξs[p])
            @test ϕᵢ_at_ξₚ ≈ (i == p ? 1.0 : 0.0) atol=1e-12
        end
    end

    @testset "Partition of unity off-node" begin
        # Σᵢ ϕᵢ(ξ) == 1 at any ξ for Lagrange basis
        for ξ in (-0.7, -0.3, 0.0, 0.42, 0.91)
            s = sum(Lagrange.evaluate(ξs, i, ξ) for i in 1:4)
            @test s ≈ 1.0 atol=1e-12
        end
    end

    @testset "evaluate_all returns a vector" begin
        ξ = 0.3
        all = Lagrange.evaluate_all(ξs, ξ)
        @test length(all) == 4
        @test sum(all) ≈ 1.0 atol=1e-12
        for i in 1:4
            @test all[i] ≈ Lagrange.evaluate(ξs, i, ξ) atol=1e-12
        end
    end

    @testset "evaluate_all! mutates the buffer" begin
        ξs = [-1.0, 0.0, 1.0]
        out = zeros(3)
        Lagrange.evaluate_all!(out, ξs, 0.5)
        @test out ≈ Lagrange.evaluate_all(ξs, 0.5)
    end
end
