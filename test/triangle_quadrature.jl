using ConservativeRegridding: TriangleQuadrature
using Test

@testset "Triangle quadrature" begin
    @testset "Reference triangle rule integrates polynomials exactly" begin
        # Reference triangle: vertices (0,0), (1,0), (0,1), area = 1/2.
        # ∫∫ x^a y^b dA = a! b! / (a+b+2)!  (standard identity)
        for degree in 1:8
            bary, w = TriangleQuadrature.reference_rule(degree)
            @test sum(w) ≈ 0.5 atol=1e-12  # weights sum to triangle area
            for a in 0:degree, b in 0:(degree - a)
                # bary points are (λ₁, λ₂, λ₃); cartesian (x,y) = (λ₂, λ₃)
                I = sum(w[k] * bary[k][2]^a * bary[k][3]^b for k in eachindex(w))
                exact = factorial(a) * factorial(b) / factorial(a + b + 2)
                @test I ≈ exact atol=1e-10
            end
        end
    end

    @testset "Fan triangulation of a square" begin
        # Square (0,0)-(1,0)-(1,1)-(0,1), centroid (0.5, 0.5)
        verts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        tris = TriangleQuadrature.fan_triangulate(verts)
        @test length(tris) == 4
        # Each triangle: (centroid, vᵢ, vᵢ₊₁); total planar area = 1.
        total = sum(TriangleQuadrature.planar_triangle_area(t) for t in tris)
        @test total ≈ 1.0 atol=1e-12
    end
end
