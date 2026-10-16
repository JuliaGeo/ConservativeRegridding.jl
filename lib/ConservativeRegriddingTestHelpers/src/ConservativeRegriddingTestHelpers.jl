"""
    ConservativeRegriddingTestHelpers

Shared helpers for the sweat tests (`test/sweat.jl`, `test/sweat_field.jl`).
"""
module ConservativeRegriddingTestHelpers

using Test
import GeometryOps as GO
import GeoInterface as GI
import IterTools
import SparseArrays
using LinearAlgebra: cross, dot, norm

import ConservativeRegridding
using ConservativeRegridding: Trees, TriangleQuadrature

import GridInterface as GrI
using GridInterface: GridAndTree
import GeometryOpsCore as GOC
import SpatialTreeInterface as STI

export has_tripolar, has_rotated, has_spectral_element, has_full_ring_grid, has_cell_crossing_dateline
export test_intersection_areas_agree, test_integration_weights, test_order_symmetry
export SphericalPolygonIntegrator, set_field_values!, zero_field!
export CRTree, cr_tree, grid_of, test_weights_agree

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
    src_tree = Trees.treeify(manifold, src_field)
    dst_tree = Trees.treeify(manifold, dst_field)
    op = ConservativeRegridding.DefaultIntersectionOperator(manifold)
    A = regridder.intersections  # rows = dst, cols = src; nonzeros are area(src ∩ dst)
    rows = SparseArrays.rowvals(A)
    vals = SparseArrays.nonzeros(A)
    worst = 0.0
    for s in axes(A, 2)  # src cell
        isempty(SparseArrays.nzrange(A, s)) && continue
        src_cell = Trees.getcell(src_tree, s)
        sa = regridder.src_areas[s]
        for k in SparseArrays.nzrange(A, s)
            d = rows[k]
            fwd = vals[k]
            denom = min(sa, regridder.dst_areas[d])
            (isfinite(fwd) && denom > 0) || continue
            rev = op(Trees.getcell(dst_tree, d), src_cell)  # reverse clip order
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


# ===========================================================================
# GridInterface: the bridge into CR's engine, the provider of a native grid,
# and the weight comparison between the native and the provider path.
# (The bridge lives here for now; it moves into CR with Phase 7.)
# ===========================================================================

"""
    CRTree(gt::GridAndTree)
    CRTree(grid, node)

STI tree that forwards traversal to `node` (the pair's tree, or a node of it,
whose leaves emit positions of the grid) and answers CR's cell-access hooks
from the grid: `getcell(t, i) = cell_geometry(grid, i)`, `cell_index_count(t) =
ncells(grid)`, `best_manifold(t) = best_manifold(grid)`, `split_weight(t) =
split_weight(node)`, `subtree_indices(t) = subtree_indices(node)`.
"""
struct CRTree{G,T}
    grid::G
    node::T
end
CRTree(gt::GridAndTree) = CRTree(gt, gt.tree)
cr_tree(gt::GridAndTree) = CRTree(gt)

"""
    cr_tree(grid) -> CRTree

`CRTree(GridAndTree(grid; leaf_capacity = ForSweeps()))`: any GridInterface
grid as a tree CR's `Regridder` consumes, `Regridder(cr_tree(dst), cr_tree(src))`.
"""
cr_tree(g) = CRTree(GridAndTree(g; leaf_capacity=GrI.ForSweeps()))

STI.isspatialtree(::Type{<:CRTree}) = true
STI.node_extent_is_expensive(::Type{CRTree{G,T}}) where {G,T} = STI.node_extent_is_expensive(T)
STI.isleaf(t::CRTree) = STI.isleaf(t.node)
STI.nchild(t::CRTree) = STI.nchild(t.node)
STI.getchild(t::CRTree) = (CRTree(t.grid, c) for c in STI.getchild(t.node))
STI.getchild(t::CRTree, i::Int) = CRTree(t.grid, STI.getchild(t.node, i))
STI.node_extent(t::CRTree) = STI.node_extent(t.node)
STI.child_indices_extents(t::CRTree) = STI.child_indices_extents(t.node)

Trees.getcell(t::CRTree, i::Int) = GrI.cell_geometry(t.grid, i)
Trees.getcell(t::CRTree) = (GrI.cell_geometry(t.grid, i) for i in 1:GrI.ncells(t.grid))
# GridInterface trees answer GrI.split_weight, CR trees Trees.split_weight (each
# defaults to 1 on the other's tree).
_sw(n) = max(GrI.split_weight(n), Trees.split_weight(n))
Trees.split_weight(t::CRTree) = _sw(t.node)
Trees.ncells(t::CRTree) = GrI.ncells(t.grid)   # the cell count (CR's SE operators size the FV side by it); not the split weight
GrI.split_weight(t::CRTree) = _sw(t.node)
GrI.subtree_indices(t::CRTree) = GrI.subtree_indices(t.node)
Trees.cell_index_count(t::CRTree) = GrI.ncells(t.grid)
GOC.best_manifold(t::CRTree) = GOC.best_manifold(t.grid)
Base.show(io::IO, t::CRTree) = print(io, "CRTree(", typeof(t.grid).name.name, ", ", t.node, ")")

# GridInterface grids (and the wrappers) go straight into Regridder(dst, src).
Trees.treeify(::GOC.Manifold, g::Union{GrI.AbstractGrid,GrI.AbstractWrappedGrid}; kw...) = cr_tree(g)

"""
    grid_of(x) -> GridInterface grid

The GridInterface provider of a native grid or field (an Oceananigans grid,
a ClimaCore cubed-sphere space or topology, a RingGrids grid, a
`Healpix.HealpixMap`), with the cells CR's native path regrids: a masked
(`ImmersedBoundaryGrid`) or U-pivot-folded Oceananigans grid answers the
`Subset` of its wet, non-duplicated cells. Defined by the test-helper
extensions for each package; a GridInterface grid is its own.
"""
grid_of(g::Union{GrI.AbstractGrid,GrI.AbstractWrappedGrid}) = g

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

`@test` that the regridder built on the provider path (`new`) has the same
sparse pattern, weights and areas as the native one (`old`): bit-identical
by default, within `rtol` when given. When the provider grid is a `Subset`
(wet or unique cells), pass it as `dst`/`src`: the native matrix is compared
on the subset's parent positions and the rows/columns outside must be zero.
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
