# Second-order conservative regridding — design

Date: 2026-04-26
Status: Design validated against ESMF source, ready for implementation
Branch context: `as/newapi`

## Motivation

ConservativeRegridding.jl currently implements first-order conservative regridding: a sparse intersection-area matrix `A[dst, src]` is multiplied by the source field and divided by destination cell areas. This conserves the integral and area-weighted mean but smooths sharp features.

Second-order conservative regridding additionally preserves the gradient of the source field via a Taylor reconstruction inside each source cell:

```
f_n(r⃗) ≈ ⟨f⟩_n + ∇_n f · (r⃗ - r⃗_n)
```

The destination value is the area-weighted mean of this reconstruction over each destination cell. The result has higher-order accuracy on smooth fields and noticeably reduces smearing of sharp features compared to 1st-order, while still being strictly conservative.

This design follows ESMF's `CONSERVE_2ND` algorithm (Earth System Modeling Framework). The gradient is reconstructed per source cell via a discrete Green's theorem on a polygon formed by the centroids of node-neighbour cells. Conservation is enforced algebraically by construction: a constant source field reproduces exactly.

The full ESMF reference is `src/Infrastructure/Mesh/src/Regridding/ESMCI_Conserve2ndInterp.C` in the [esmf-org/esmf](https://github.com/esmf-org/esmf) repository. Algorithm-level cross-checking against that source is summarized inline below; design choices that diverge are flagged.

## Decisions (summary)

1. **API**: keep a single `Regridder` struct. Add an `algorithm::A` field where `A <: RegriddingAlgorithm`. Construct 2nd-order via `Regridder(dst, src; algorithm = SecondOrderConservative())`. No separate `SecondOrderRegridder` type.
2. **Capabilities** are queried on the algorithm, not the regridder. `supports_transpose(::FirstOrderConservative) = true`, `supports_transpose(::SecondOrderConservative) = false`. `Base.transpose(r::Regridder)` dispatches via a 2-arg method `Base.transpose(r::Regridder, ::FirstOrderConservative)`; the absence of a method for `SecondOrderConservative` produces a `MethodError`.
3. **Centroid algorithm** is a field on `SecondOrderConservative`, configurable via abstract `CentroidAlgorithm`. Two concrete types: `ESMFLike` (default; mimics ESMF: vertex-average projected to sphere for neighbours, area-weighted from supermesh for source-cell centroid) and `GeometricCentroid` (uses `GO.centroid_and_area` for both).
4. **Number type** is a type parameter `T <: AbstractFloat`, default `Float64`, set via `Regridder(...; T = Float32)`.
5. **Neighbour stencil** reuses the existing `Trees.neighbours` (node-neighbours, 1-ring of 8 cells for an interior quad, falls out to 7 at cube corners). v1 is restricted to grids that implement it: cubed sphere via `CubedSphereToplevelTree` and lon-lat via `LonLatConnectivityWrapper`. Other grid types raise an `ArgumentError` at construction.
6. **No mask support** in v1.
7. **Apply path** stays a single concrete method on `Regridder` — the algorithm is already encoded in the matrix at construction time, so `regrid!` is independent of order.
8. **All implementation in a single new file** `src/regridder/second_order.jl`. The existing 1st-order files are touched minimally: `regridder.jl` for the type-parameter and field additions; `intersection_areas.jl` for the algorithm-dispatched per-pair kernel.

## Architecture

### Types

```julia
abstract type RegriddingAlgorithm end

struct FirstOrderConservative <: RegriddingAlgorithm end

abstract type CentroidAlgorithm end

"ESMF-style: area-weighted from supermesh for the source-cell centroid (the
Taylor expansion point); vertex-average projected to the sphere for neighbour
centroids (the gradient stencil)."
struct ESMFLike <: CentroidAlgorithm end

"Use `GO.centroid_and_area(manifold, polygon)` for both source and neighbour
centroids. More accurate for curved cells, slightly slower."
struct GeometricCentroid <: CentroidAlgorithm end

struct SecondOrderConservative{C <: CentroidAlgorithm} <: RegriddingAlgorithm
    centroid_algorithm::C
end
SecondOrderConservative() = SecondOrderConservative(ESMFLike())

abstract type AbstractRegridder end

struct Regridder{T <: AbstractFloat, A <: RegriddingAlgorithm, ...} <: AbstractRegridder
    intersections::SparseMatrixCSC{T, Int}
    dst_areas::Vector{T}
    src_areas::Vector{T}
    algorithm::A
    tmp::Vector{T}
    # ... existing fields preserved
end
```

### Constructor

```julia
function Regridder(dst, src; algorithm = FirstOrderConservative(), T = Float64)
    # 1) treeify dst and src — existing
    # 2) multithreaded_dual_query — existing — yields candidate (dst, src) pairs
    # 3) build_weights(algorithm, manifold, dst_tree, src_tree, T) → SparseMatrixCSC{T}
    # 4) compute dst_areas, src_areas — existing
    # 5) wrap in Regridder
end
```

`build_weights` is multiple-dispatched on the algorithm. The 1st-order method is the existing path. The 2nd-order method lives in `second_order.jl`.

### Capabilities

```julia
supports_transpose(::RegriddingAlgorithm) = false        # safe default
supports_transpose(::FirstOrderConservative) = true

Base.transpose(r::Regridder) = transpose(r, r.algorithm)
function Base.transpose(r::Regridder, ::FirstOrderConservative)
    # the existing swap (src ↔ dst, return new Regridder sharing data)
end
# No method for SecondOrderConservative → automatic MethodError
```

### Apply path

Unchanged from today, single concrete method on `Regridder`:

```julia
function regrid!(out, r::Regridder, src)
    mul!(r.tmp, r.intersections, src)
    out .= r.tmp ./ r.dst_areas
    return out
end
```

The 2nd-order weights have already absorbed all gradient terms into `r.intersections`, so `regrid!` does no extra work.

## Geometry pipeline

Three phases. Phases 1–2 reuse existing infrastructure with minor algorithm-dispatched changes; Phase 3 is new and 2nd-order-only.

### Phase 1 — candidate pair search (shared)

`multithreaded_dual_query(dst_tree, src_tree)` walks both spatial trees in lockstep, pruning by extent. Output: candidate `(dst_idx, src_idx)` pairs that *might* intersect. Unchanged.

### Phase 2 — per-pair intersection geometry (algorithm-dispatched)

The per-chunk loop calls a kernel:

```julia
intersection_value(alg, manifold, src_poly, dst_poly) -> Tuple

# 1st-order: returns area only — equivalent to current code.
intersection_value(::FirstOrderConservative, manifold, src_poly, dst_poly) = (area,)

# 2nd-order: returns area and centroid c_nk.
function intersection_value(alg::SecondOrderConservative, manifold, src_poly, dst_poly)
    poly = compute_intersection_polygon(manifold, src_poly, dst_poly)
    area = polygon_area(manifold, poly)
    c_nk = sm_cell_centroid(alg.centroid_algorithm, manifold, poly)
    return (area, c_nk)
end

sm_cell_centroid(::ESMFLike,         manifold::Spherical, poly) = normalize(mean(vertices(poly)))
sm_cell_centroid(::ESMFLike,         manifold::Planar,    poly) = mean(vertices(poly))
sm_cell_centroid(::GeometricCentroid, manifold,           poly) = first(GO.centroid_and_area(manifold, poly))
```

The centroid type is `SVector{3,T}` (spherical) or `SVector{2,T}` (planar).

For 1st-order, the chunked output `(I, J, V::Vector{Float64})` flows directly into `SparseArrays.sparse(I, J, V, ndst, nsrc, +)` exactly as today.

For 2nd-order, the chunked output is a `Vector{NamedTuple{(:dst,:src,:area,:cntr)}}`. This is the input to Phase 3.

### Phase 3 — gradient + weight emission per source cell (2nd-order only)

```
# 1) Group sm_cells by src_idx into a CSR-style index:
counts = zeros(Int, n_src)
for cell in sm: counts[cell.src] += 1
offsets = cumsum(counts)                  # [end] = total
sm_per_src = Vector{SMCell}(undef, length(sm))
write_pos = copy(offsets) .- counts
for cell in sm:
    write_pos[cell.src] += 1
    sm_per_src[write_pos[cell.src]] = cell
src_areas[n] = Σ over sm_per_src_n of cell.area    # cached

# 2) Parallelise over src cells in nthreads*4 chunks:
@threads for chunk in src_chunks
    local_I, local_J, local_V = Int[], Int[], T[]
    for n in chunk:
        sm_cells_n = view(sm_per_src, offsets[n]-counts[n]+1 : offsets[n])
        isempty(sm_cells_n) && continue
        process_source_cell!(local_I, local_J, local_V,
                             algorithm, manifold, src_tree, n, sm_cells_n,
                             dst_areas)
    end
end
# concat thread-local triples → assemble:
W = SparseArrays.sparse(I, J, V, n_dst, n_src, +)   # + merges duplicate (I,J)
```

Per-source-cell processing (`process_source_cell!`) is described in detail in the next section.

## Gradient algorithm (per source cell)

Implemented inside `second_order.jl` as `process_source_cell!`. Consumes `sm_cells_n`, `n` (source index), `src_tree`, `algorithm`, `manifold`, `dst_areas`. Pushes to thread-local `(I, J, V)`.

### 1. Source centroid r_n

```
ESMFLike, Spherical:        r_n = normalize(Σ_k A_nk · c_nk)
ESMFLike, Planar:           r_n = (Σ_k A_nk · c_nk) / Σ_k A_nk
GeometricCentroid, both:    r_n, _ = GO.centroid_and_area(manifold, Trees.getcell(src_tree, n))
```

Cache `u_src_cntr = r_n / |r_n|` for use in the angle formula and tangent projection (these are sensitive to non-unit length on the spherical path).

### 2. Neighbour list and centroids

```
nbr_ids = Trees.neighbours(src_tree, n)          # node-neighbour 1-ring; 8 cells typically, 7 at cube corners
for m in nbr_ids:
    poly_m = Trees.getcell(src_tree, m)
    ESMFLike, Spherical:        c_m = normalize(mean(vertices(poly_m)))
    ESMFLike, Planar:           c_m = mean(vertices(poly_m))
    GeometricCentroid, both:    c_m, _ = GO.centroid_and_area(manifold, poly_m)
```

### 3. CCW sort around r_n

The reference neighbour is the one with the largest id, *excluding* any whose centroid coincides exactly with `r_n`:

```
ref = argmax(m for m in nbr_ids if c_m != r_n; init = 0)
if ref == 0: zero_gradient_fallback(); return
v1  = c[ref] - r_n
for each m in nbr_ids:
    angle_m = ccw_angle(v1, c_m - r_n; normal = u_src_cntr)
sort (nbr_ids, c) jointly by angle_m
```

`ccw_angle` formulas (from `ESMCI_MathUtil.C:3166-3183`):
- Spherical: `atan2((a × b) · n̂, a · b)` with `n̂ = u_src_cntr`.
- Planar: `atan2(a[1] b[2] - a[2] b[1], a · b)` (the normal arg is unused).

### 4. Fallbacks (zero-fill all gradient vectors and continue to weight emission)

(a) `length(nbr_ids) < 3`
(b) `length(nbr_ids) > 150` (matches ESMF's `MAX_NUM_NBRS`; ESMF errors, we fall back)
(c) `r_n` not inside `polygon([c_m...])` (point-in-polygon: `is_pnt_in_polygon` per manifold, tolerance `1e-14`)
(d) `nbr_poly_area == 0`
(e) Any consecutive-neighbour pair `(c[i], c[prev])` is antipodal: `c[i] × c[prev] ≈ 0` AND `c[i] · c[prev] < 0`
(f) Reference-neighbour selection failed (no neighbour with `c_m ≠ r_n`)

If `c[i] × c[prev] ≈ 0` with `c[i] · c[prev] > 0` (parallel, not antipodal), skip that edge in the Green's-theorem loop but don't fall back.

### 5. Green's-theorem gradient weights

```
nbr_poly_area = polygon_area(manifold, c)
grad_per_nbr  = zeros(SVector{3,T} or SVector{2,T}, length(nbrs))
src_grad      = zero(SVector{3,T} or SVector{2,T})

for i in 1:N:
    prev = mod1(i - 1, N)
    Spherical:
        n̂ = c[i] × c[prev]
        if |n̂| ≈ 0: handle per fallback (e); skip-or-zero-out
        arc_len = acos(clamp(c[i] · c[prev], -1, 1))
        contrib = (n̂ / |n̂|) * arc_len
    Planar:
        edge    = c[i] - c[prev]
        contrib = SVector(edge[2], -edge[1])           # 90° CW rotation; outward normal for CCW polygon
    grad_per_nbr[i]    += contrib
    grad_per_nbr[prev] += contrib
    src_grad           += contrib
```

**Asymmetric scaling** (this is the conservation invariant — verified against ESMF lines 988–1000):

```
for g in grad_per_nbr:
    Spherical: g = (u_src_cntr × g) × u_src_cntr     # tangent projection at r_n
    g /= 2 * nbr_poly_area
Spherical: src_grad = (u_src_cntr × src_grad) × u_src_cntr
src_grad /= nbr_poly_area                             # NOT 2× — this is the fix
```

The factor-of-2 difference between `grad_per_nbr` (÷ 2A) and `src_grad` (÷ A) is essential. It cancels the 2× over-counting in the Green's loop (each edge contributes to two neighbours' grads but only once to `src_grad`'s pre-scaling sum) and makes `Σ_m grad_m_scaled = src_grad_scaled`, which is what makes constant fields exactly preserved at weight emission time.

### 6. Weight emission per (n, k)

```
for sm_cell in sm_cells_n:
    diff  = sm_cell.cntr - r_n
    ratio = sm_cell.area / dst_areas[sm_cell.dst]
    if 1.0 < ratio < 1.0 + 1e-10: ratio = 1.0    # ESMF round-off clamp
    push (sm_cell.dst, n, ratio - (diff · src_grad) * ratio) to (I, J, V)
    for (m, g_m) in zip(nbr_ids, grad_per_nbr):
        push (sm_cell.dst, m, (diff · g_m) * ratio)
```

After all source cells: `SparseArrays.sparse(I, J, V, n_dst, n_src, +)` merges duplicates.

**Conservation invariant** (verified algebraically): for `f ≡ c` (constant), the destination value is

```
F_k = Σ_n [ratio_nk * (c - diff_nk · src_grad_n · c)
         + Σ_m diff_nk · grad_n,m * c]
    = c * Σ_n ratio_nk * (1 - diff_nk · src_grad_n + diff_nk · Σ_m grad_n,m)
    = c * Σ_n ratio_nk                                 # since Σ_m grad_n,m = src_grad_n
    = c * (1 / dst_area_k) * Σ_n A_nk
    = c                                                # since Σ_n A_nk = dst_area_k
```

## Testing strategy

Four new test files under `test/regridder/` and one at `test/xesmf_comparison.jl`. All wired into `test/runtests.jl` via `@safetestset`.

### `test/regridder/second_order_lonlat.jl`

Lon-lat → lon-lat regridding (e.g. 36×18 → 24×12, both global periodic with pole folds).

1. **Constant-field exactness** with `f ≡ 7.0`: `dst .≈ 7.0` to `rtol = 1e-12`. Proves the asymmetric `÷ A` vs `÷ 2A` scaling.
2. **Linear-field reproduction**: regrid `f(λ, φ) = a + b·λ + c·φ` (away from poles), assert recovery to `rtol = 1e-6`. Centroid-rule error is order h².
3. **Conservation under non-trivial source**: `f = cos(λ) sin(2φ)`, assert `Σ dst .* dst_areas ≈ Σ src .* src_areas` to `rtol = 1e-12`.
4. **API smoke tests**:
   - `Regridder(dst, src; algorithm = SecondOrderConservative())` builds.
   - `r.algorithm isa SecondOrderConservative` and `r.algorithm.centroid_algorithm isa ESMFLike`.
   - `transpose(r)` raises `MethodError`.
   - `supports_transpose(r.algorithm) === false`.
5. **Number-type kwarg**: `Regridder(...; T = Float32)` returns `Regridder{Float32, ...}`, regrid works.
6. **`GeometricCentroid` algorithm path**: `Regridder(...; algorithm = SecondOrderConservative(GeometricCentroid()))` builds, gives results close to `ESMFLike` (rtol = 1e-3, since the two centroid choices give different weights but both should regrid a smooth field similarly).

### `test/regridder/second_order_cubed_sphere.jl`

Cubed-sphere → lon-lat and lon-lat → cubed-sphere.

1. Constant-field exactness across cube-edge crossings.
2. Conservation: `Σ dst .* dst_areas ≈ Σ src .* src_areas` to `rtol = 1e-10`.
3. **Cube-edge stencil correctness**: pick a destination cell that overlaps source cells *across* a cube edge. Inspect `r.intersections[k, :]`; assert there are nonzero entries at column indices corresponding to neighbour cells on the *adjacent face* (not just the source face). Confirms `Trees.neighbours` is supplying cross-face neighbours and the gradient is using them.
4. **Order-of-accuracy convergence**: regrid a smooth field (e.g. spherical harmonic `Y_2^1`) at two resolutions, assert L2 error scales as `O(h²)` for 2nd-order vs `O(h)` for 1st-order. Soft-asserted (`@test_skip` if noisy).

### `test/regridder/second_order_unit.jl`

Fine-grained unit tests on the inner kernels. Construct synthetic neighbour stencils and source centroids; do not require a full regridder build.

1. **Green's gradient on a known stencil**: hand-build a 4×4 lon-lat patch, manually compute expected `∇f` for a linear field, assert per-source-cell gradient matches to `rtol = 1e-10`.
2. **CCW sort**: 5 random neighbour centroids around a known source centroid, assert the sorted order is monotonic CCW.
3. **Each fallback path produces zero gradient**:
   - 2 neighbours
   - 151 neighbours
   - antipodal-neighbour synthetic stencil
   - `r_n` outside `polygon(c_m)` synthetic stencil
   - all neighbours coincide with `r_n` (reference selection fails)
4. **Round-off clamp**: synthetic pair where `A_nk = dst_area * (1 + 5e-11)` clamps to 1.0.

### `test/xesmf_comparison.jl`

ESMF reference comparison. Detailed plan being produced by a separate research pass and will land at `docs/plans/2026-04-26-xesmf-comparison-plan.md`. Skeleton:

- Tests gracefully `@test_skip` when xesmf/ESMPy is not installed.
- Three grid pairs: lon-lat global, lon-lat regional, cubed-sphere → lon-lat.
- For each pair × each method (1st-order, 2nd-order):
  - Compute reference weights via Python.
  - Build our `Regridder` with the same configuration.
  - Compare matrices: `nnz`, max-abs-diff per nonzero, row sums.
  - Apply both to a known field, compare results (rtol = 1e-10 for 1st-order, 1e-8 for 2nd-order; loosen if Python's intersection algorithm differs from ours).

## File-by-file change list

### New

- `src/regridder/second_order.jl`
  - `RegriddingAlgorithm`, `FirstOrderConservative`, `SecondOrderConservative{C}`.
  - `CentroidAlgorithm`, `ESMFLike`, `GeometricCentroid`.
  - `supports_transpose` and `Base.transpose(r::Regridder, ::FirstOrderConservative)`.
  - `intersection_value(::SecondOrderConservative, ...)` returning `(area, c_nk)`.
  - `sm_cell_centroid`, `source_centroid`, `neighbour_centroid`, `sort_ccw!`, `green_gradient!`, `process_source_cell!`.
  - `build_weights(::SecondOrderConservative, ...)` — the Phase-3 driver.
- `test/regridder/second_order_lonlat.jl`
- `test/regridder/second_order_cubed_sphere.jl`
- `test/regridder/second_order_unit.jl`
- `test/xesmf_comparison.jl` (per follow-up plan)

### Modified

- `src/regridder/regridder.jl`
  - Introduce `AbstractRegridder` supertype.
  - Add `T <: AbstractFloat` and `A <: RegriddingAlgorithm` type parameters to `Regridder`.
  - Add `algorithm::A` field.
  - Constructors take `algorithm = FirstOrderConservative()` and `T = Float64` kwargs.
- `src/regridder/regrid.jl`
  - No functional change. `regrid!` stays defined on the concrete `Regridder`. Algorithm is encoded in the matrix.
- `src/regridder/intersection_areas.jl`
  - Refactor the per-pair geometry kernel to dispatch via `intersection_value(algorithm, ...)`. The 1st-order path is the existing code, now wrapped behind that dispatch. The 2nd-order method is in `second_order.jl`.
- `src/ConservativeRegridding.jl`
  - `include("regridder/second_order.jl")` after the existing regridder includes.
  - Decide which symbols are `export` vs `@public`. Leaning: `export FirstOrderConservative, SecondOrderConservative`; `@public ESMFLike, GeometricCentroid, supports_transpose`.
- `test/runtests.jl`
  - `@safetestset` blocks for the four new files. The xesmf one wrapped in skip-on-missing-Python.

### Not modified

- `src/trees/*` — `Trees.neighbours` and `Trees.getcell` are consumed via existing API.
- All `ext/*` extension files — algorithm-agnostic.
- `src/utils/MultithreadedDualDepthFirstSearch.jl` — reused as-is.
- `src/trees/neighbours_interface.jl` — interface stubs untouched.

## Public API summary

```julia
using ConservativeRegridding

# 1st-order (unchanged behaviour):
r1 = Regridder(dst, src)                                                    # default algorithm
regrid!(out, r1, field)
transpose(r1)                                                                # works

# 2nd-order:
r2 = Regridder(dst, src; algorithm = SecondOrderConservative())             # ESMFLike default
r2 = Regridder(dst, src; algorithm = SecondOrderConservative(GeometricCentroid()))
r2 = Regridder(dst, src; algorithm = SecondOrderConservative(), T = Float32)
regrid!(out, r2, field)
transpose(r2)                                                                # MethodError
ConservativeRegridding.supports_transpose(r2.algorithm)                      # false
```

## Deferred / out of scope for this round

- Mask support for both 1st-order and 2nd-order.
- ESMF orientation transform across cube edges (not needed for our scalar 2nd-order; would matter for vector fields).
- Other 2nd-order grid types beyond cubed sphere and lon-lat (HEALPix, Oceananigans grids, SpeedyWeather grids). Each needs a `Trees.neighbours` implementation first.
- Monotonicity limiters / gradient clipping. ESMF treats these as a separate code path (`--mono`); we leave them to a follow-up.
- Distributed / MPI parallelism. Our threading is intra-node only.
- A `findidx` and `dual_neighbours` implementation for non-conservative interpolation. Already deferred from the neighbours-interface design.

## References

- ESMF source (master), specifically:
  - `src/Infrastructure/Mesh/src/Regridding/ESMCI_Conserve2ndInterp.C`
  - `src/Infrastructure/Mesh/include/Regridding/ESMCI_Conserve2ndInterp.h`
  - `src/Infrastructure/Mesh/src/Legacy/ESMCI_SM.C` (supermesh cell creation, sm_cell centroid)
  - `src/Infrastructure/Mesh/src/Regridding/ESMCI_Interp.C` (driver: `calc_2nd_order_conserve_mat_serial_*`)
  - `src/Infrastructure/Mesh/src/ESMCI_MathUtil.C` (angle, cross product, orth-2D)
- ESMF `Regrid_implnotes.tex` (lat/lon line-integral formulation; older notes — the actual code uses the 3D-vector Green's-theorem formulation summarised here).
- Jones (1999) for 1st-order conservative remapping foundations.
- Sister design: `docs/plans/2026-04-26-neighbours-interface-design.md` (the `Trees.neighbours` interface this design depends on).
- Follow-up plan: `docs/plans/2026-04-26-xesmf-comparison-plan.md` (xesmf/ESMPy reference-comparison test design).
