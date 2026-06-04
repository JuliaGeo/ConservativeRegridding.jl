# Shared helpers for the sweat tests: `sweat.jl` (vector path, `regrid!` on raw
# value vectors) and `sweat_field.jl` (field path, `regrid!` on package-native
# fields). Include this *after* the includer has run `using Test`, imported
# `GeometryOps as GO` / `GeoInterface as GI` / `ClimaCore`, brought
# `ConservativeRegridding`'s `Trees` into scope, and defined the `ClimaCoreExt`
# extension handle — the helpers below resolve those names from the caller.

import SimplexQuad
import IterTools
using LinearAlgebra: cross, dot, norm

# The intersection matrix reproduces the geometric cell areas on both sides.
function test_intersection_areas_agree(regridder, tree1, tree2; rtol = sqrt(eps(Float64)))
    @test sum(regridder.intersections, dims=2)[:, 1] ≈ regridder.dst_areas rtol=rtol
    @test sum(regridder.intersections, dims=1)[1, :] ≈ regridder.src_areas rtol=rtol
end

# Quadrature over a spherical polygon: fan-triangulate from the first vertex and
# integrate each great-circle triangle on the unit sphere.
struct SphericalPolygonIntegrator{X, W}
    x::X
    w::W
    function SphericalPolygonIntegrator(; order=7)
        X, W = SimplexQuad.simplexquad(order, 2)
        new{typeof(X), typeof(W)}(X, W)
    end
end

function (integrator::SphericalPolygonIntegrator)(vertices::AbstractVector{<:GO.UnitSpherical.UnitSphericalPoint}, f)
    X, W = integrator.x, integrator.w
    total = 0.0
    A = vertices[1]
    for i in 2:length(vertices)-1
        B, C = vertices[i], vertices[i+1]
        det_ABC = dot(A, cross(B, C))
        for k in axes(X, 1)
            ξ1, ξ2 = X[k, 1], X[k, 2]
            ξ0 = 1 - ξ1 - ξ2
            p = ξ1 * A + ξ2 * B + ξ0 * C
            np = norm(p)
            s = p / np
            J = abs(det_ABC) / np^3
            total += W[k] * f(s) * J
        end
    end
    return total
end

# FV cell value = area-weighted mean of `fun`, i.e. `∫f dΩ / ∫dΩ`. Numerator and
# the solid-angle normalizer share the same unit-sphere integrator, so the result
# is independent of the manifold radius (dividing by the physical area instead
# would leave a stray `1/R²` factor). Zero-solid-angle cells are folded-grid
# ghost partners (e.g. Oceananigans `RightCenterFolded` tripolar); they
# contribute nothing, so yield 0 instead of `0/0 = NaN`.
function set_field_values!(field, values, fun; integrator = SphericalPolygonIntegrator(; order=7))
    tree = Trees.treeify(field)
    polys = IterTools.ivec(Trees.getcell(tree))
    values .= Iterators.map(polys) do poly
        points = GI.getpoint(GI.getexterior(poly))
        solid_angle = integrator(points, _ -> 1.0)
        iszero(solid_angle) ? zero(eltype(values)) :
            integrator(points, p -> fun((GO.UnitSpherical.GeographicFromUnitSphere()(p))...)) / solid_angle
    end
end

# ClimaCore fields store one value per spectral-element node; the principled SE
# regridder consumes/produces those nodal values directly. Sample `fun` at each
# node position instead of integrating over a cell polygon.
function set_field_values!(field::ClimaCore.Fields.Field, values, fun; kwargs...)
    positions = ClimaCoreExt.se_node_positions(getfield(field, :space))
    values .= Iterators.map(positions) do p
        fun((GO.UnitSpherical.GeographicFromUnitSphere()(p))...)
    end
end

function zero_field!(field, values)
    set_field_values!(field, values, (x, y, z = 0) -> 0)
end

# Integration weights for the conservation check: cell areas for FV grids,
# per-node quadrature weights (`WJ`) for SE fields, so `sum(vals .* weights)` is
# the sphere integral in both cases.
test_integration_weights(field, regridder) = regridder.dst_areas
test_integration_weights(field::ClimaCore.Fields.Field, regridder) =
    ClimaCoreExt.se_node_weights(getfield(field, :space))
