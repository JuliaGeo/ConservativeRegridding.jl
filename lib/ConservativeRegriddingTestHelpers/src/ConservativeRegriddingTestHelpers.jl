"""
    ConservativeRegriddingTestHelpers

Shared helpers for the sweat tests (`test/sweat.jl`, `test/sweat_field.jl`)
and the provider tests (`test/providers.jl`).
"""
module ConservativeRegriddingTestHelpers

using Test
import GeometryOps as GO
import GeoInterface as GI
import SparseArrays
using LinearAlgebra: cross, dot, norm

import ConservativeRegridding
using ConservativeRegridding: TriangleQuadrature, asgrid

import GridInterface as GrI
import GeometryOpsCore as GOC

export has_tripolar, has_rotated, has_spectral_element, has_full_ring_grid, has_cell_crossing_dateline
export test_intersection_areas_agree, test_integration_weights, test_order_symmetry
export SphericalPolygonIntegrator, set_field_values!, zero_field!
export grid_of, test_weights_agree

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
    has_full_ring_grid(field)

True if `field` is on a full RingGrids grid (e.g. SpeedyWeather's FullClenshaw /
FullGaussian), whose first ring cell is centered at lon 0° and spans the 0°/360° seam.
"""
has_full_ring_grid(field) = false

"""
    has_cell_crossing_dateline(field)

True if `field` has cells that cross the 0°/360° longitude seam (the "dateline"):
tripolar and rotated lat-lon grids (near their poles), full RingGrids grids, and
cubed-sphere SE panels. Such grids smear the discontinuous `LongitudeField` by ~1%
under regridding, so the sweat tests loosen the conservation tolerance for them.
"""
has_cell_crossing_dateline(field) =
    has_tripolar(field) || has_rotated(field) || has_spectral_element(field) || has_full_ring_grid(field)

"""
    test_intersection_areas_agree(regridder, tree1, tree2; rtol = sqrt(eps(Float64)))

`@test` that the intersection matrix reproduces the geometric cell areas on both sides.
"""
function test_intersection_areas_agree(regridder, tree1, tree2; rtol = sqrt(eps(Float64)))
    @test sum(regridder.intersections, dims=2)[:, 1] ≈ regridder.dst_areas rtol=rtol
    @test sum(regridder.intersections, dims=1)[1, :] ≈ regridder.src_areas rtol=rtol
end

"""
    test_order_symmetry(regridder, src_field, dst_field;
                        rtol = sqrt(eps(Float64)), manifold = GO.Spherical())

`@test` that the clipper returns the same intersection area with the source and
destination cells in either role: for every stored pair, the forward
`area(src ∩ dst)` and the reverse `area(dst ∩ src)` agree to within `rtol` of the
smaller cell's area.

Reuses the regridder's forward areas (requires `normalize = false`) and
recomputes only the reverse order. Skips non-finite and zero-area (ghost) cells.
"""
function test_order_symmetry(regridder, src_field, dst_field;
                             rtol = sqrt(eps(Float64)), manifold = GO.Spherical())
    src_grid = asgrid(manifold, src_field)
    dst_grid = asgrid(manifold, dst_field)
    op = ConservativeRegridding.DefaultIntersectionOperator(manifold, src_grid, dst_grid)
    A = regridder.intersections  # rows = dst, cols = src; nonzeros are area(src ∩ dst)
    rows = SparseArrays.rowvals(A)
    vals = SparseArrays.nonzeros(A)
    worst = 0.0
    for s in axes(A, 2)  # src cell
        isempty(SparseArrays.nzrange(A, s)) && continue
        src_cell = GrI.cell_geometry(src_grid, s)
        sa = regridder.src_areas[s]
        for k in SparseArrays.nzrange(A, s)
            d = rows[k]
            fwd = vals[k]
            denom = min(sa, regridder.dst_areas[d])
            (isfinite(fwd) && denom > 0) || continue
            rev = op(GrI.cell_geometry(dst_grid, d), src_cell)  # reverse clip order
            isfinite(rev) || continue
            worst = max(worst, abs(fwd - rev) / denom)
        end
    end
    @test worst < rtol
    return worst
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
    grid = asgrid(field)
    polys = (GrI.cell_geometry(grid, i) for i in 1:GrI.ncells(grid))
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


# ===========================================================================
# GridInterface: the provider of a native grid, and the weight comparison
# between two regridders (the native path and the provider path).
# ===========================================================================

"""
    grid_of(x) -> GridInterface grid

The GridInterface grid CR regrids `x` as: `ConservativeRegridding.asgrid(x)`
(the provider of a native grid or field, the grid itself for a GridInterface
grid). What `Regridder(x, ...)` regrids.
"""
grid_of(x) = asgrid(x)

# the old (native) matrix restricted to the subset's parent positions
function _restrict(R, rows, cols)
    A = R.intersections; da = R.dst_areas; sa = R.src_areas
    rows === nothing || (A = A[rows, :]; da = da[rows])
    cols === nothing || (A = A[:, cols]; sa = sa[cols])
    return A, da, sa
end
_outside_is_zero(A, rows, ::Val{1}) = rows === nothing || SparseArrays.nnz(A[setdiff(1:size(A, 1), rows), :]) == 0
_outside_is_zero(A, cols, ::Val{2}) = cols === nothing || SparseArrays.nnz(A[:, setdiff(1:size(A, 2), cols)]) == 0
_positions(g) = g isa GrI.Subset ? collect(GrI.parentpositions(g)) : nothing

"""
    test_weights_agree(old::Regridder, new::Regridder; rtol = 0, dst = nothing, src = nothing)

`@test` that the regridder `new` has the same sparse pattern, weights and
areas as `old`: bit-identical by default, within `rtol` when given. When
`new`'s grid is a `Subset` of `old`'s (wet or unique cells), pass it as
`dst`/`src`: the `old` matrix is compared on the subset's parent positions
and the rows/columns outside must be zero.
"""
function test_weights_agree(old, new; rtol::Real=0, dst=nothing, src=nothing)
    rows = _positions(dst); cols = _positions(src)
    A, da, sa = _restrict(old, rows, cols)
    B = new.intersections
    @test size(A) == size(B)
    @test _outside_is_zero(old.intersections, rows, Val(1))
    @test _outside_is_zero(old.intersections, cols, Val(2))
    size(A) == size(B) || return nothing
    if rtol == 0
        @test A == B
        @test new.dst_areas == da
        @test new.src_areas == sa
    else
        @test SparseArrays.findnz(A)[1:2] == SparseArrays.findnz(B)[1:2]
        @test isapprox(SparseArrays.nonzeros(A), SparseArrays.nonzeros(B); rtol)
        @test isapprox(new.dst_areas, da; rtol) && isapprox(new.src_areas, sa; rtol)
    end
    return nothing
end

end # module ConservativeRegriddingTestHelpers
