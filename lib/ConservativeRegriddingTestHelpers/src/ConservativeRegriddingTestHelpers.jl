"""
    ConservativeRegriddingTestHelpers

Shared helpers for the sweat tests (`test/sweat.jl`, `test/sweat_field.jl`).
"""
module ConservativeRegriddingTestHelpers

using Test
import GeometryOps as GO
import GeoInterface as GI
import IterTools
using LinearAlgebra: cross, dot, norm

import ConservativeRegridding
using ConservativeRegridding: Trees, TriangleQuadrature


export has_tripolar, has_rotated, has_spectral_element
export test_intersection_areas_agree, test_integration_weights
export SphericalPolygonIntegrator, set_field_values!, zero_field!

"""
    has_spectral_element(field)

True if `field` is a spectral-element field (a la ClimaCore), whose L2/principled matrices aren't pure
overlap matrices → skip the area-agreement check.
"""
has_spectral_element(field) = false

"""
    has_tripolar(field)

True if `field` is on an tripolar grid, which doesn't cover the globe
→ area sums ≠ full sphere.
"""
has_tripolar(field) = false

"""
    has_rotated(field)

True if `field` is on an rotated lat-lon grid, whose oblique cell
edges give larger area-sum error for longitude-valued field functions → loosen the tolerance.
"""
has_rotated(field) = false

"""
    test_intersection_areas_agree(regridder, tree1, tree2; rtol = sqrt(eps(Float64)))

`@test` that the intersection matrix reproduces the geometric cell areas on both sides.
"""
function test_intersection_areas_agree(regridder, tree1, tree2; rtol = sqrt(eps(Float64)))
    @test sum(regridder.intersections, dims=2)[:, 1] ≈ regridder.dst_areas rtol=rtol
    @test sum(regridder.intersections, dims=1)[1, :] ≈ regridder.src_areas rtol=rtol
end

"""
    SphericalPolygonIntegrator(; degree = 8)
    (integrator::SphericalPolygonIntegrator)(vertices, f)

Integrate `f` over a spherical polygon by fan-triangulating from vertex 1 and
integrating each great-circle triangle. Uses the symmetric reference rule (Σw=1/2)
from `ConservativeRegridding.TriangleQuadrature`, so the (λ₁,λ₂,λ₃)→(A,B,C)
assignment doesn't change the sum.
"""
struct SphericalPolygonIntegrator{B, W}
    bary::B
    w::W
    function SphericalPolygonIntegrator(; degree = 8)
        bary, w = TriangleQuadrature.reference_rule(degree)
        new{typeof(bary), typeof(w)}(bary, w)
    end
end

function (integrator::SphericalPolygonIntegrator)(vertices::AbstractVector{<:GO.UnitSpherical.UnitSphericalPoint}, f)
    bary, W = integrator.bary, integrator.w
    total = 0.0
    A = vertices[1]
    for i in 2:length(vertices)-1
        B, C = vertices[i], vertices[i+1]
        det_ABC = dot(A, cross(B, C))
        for k in eachindex(W)
            λ1, λ2, λ3 = bary[k]
            p = λ1 * A + λ2 * B + λ3 * C
            np = norm(p)
            s = p / np
            J = abs(det_ABC) / np^3
            total += W[k] * f(s) * J
        end
    end
    return total
end

"""
    set_field_values!(field, values, fun; integrator = SphericalPolygonIntegrator())

Write into `values` the FV cell-averaged `fun`, i.e. `∫f dΩ / ∫dΩ` (radius-
independent: numerator and normalizer share `integrator`). Zero-solid-angle cells
(folded-grid ghosts) yield 0 instead of `0/0 = NaN`.
"""
function set_field_values!(field, values, fun; integrator = SphericalPolygonIntegrator(; degree = 8))
    tree = Trees.treeify(field)
    polys = IterTools.ivec(Trees.getcell(tree))
    values .= Iterators.map(polys) do poly
        points = GI.getpoint(GI.getexterior(poly))
        solid_angle = integrator(points, _ -> 1.0)
        iszero(solid_angle) ? zero(eltype(values)) :
            integrator(points, p -> fun((GO.UnitSpherical.GeographicFromUnitSphere()(p))...)) / solid_angle
    end
end

"""
    zero_field!(field, values)

Zero out `values` via `set_field_values!` with a constant-zero function.
"""
function zero_field!(field, values)
    set_field_values!(field, values, (x, y, z = 0) -> 0)
end

"""
    test_integration_weights(field, regridder)

Weights for the conservation check: cell areas (FV grids) or per-node quadrature
weights (SE fields), so `sum(vals .* weights)` is the sphere integral either way.
"""
test_integration_weights(field, regridder) = regridder.dst_areas

end # module ConservativeRegriddingTestHelpers
