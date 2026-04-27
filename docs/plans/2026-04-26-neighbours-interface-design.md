# Neighbours interface — design for cubed-sphere and lon-lat grids

Date: 2026-04-26
Status: Design validated, ready for implementation
Branch context: `as/newapi`

## Motivation

`src/trees/neighbours_interface.jl` (added in commit `06ebd3c` "delineate api") declares a forward-looking API:

```julia
function findidx end                  # (grid, pos)::idx
function neighbours end               # (grid, idx)::idxs
function dual_neighbours end          # (grid, pos, neighbours)::idxs
function has_optimized_idx_search end # (grid)::Bool
function has_optimized_neighbour_search end
abstract type AbstractNeighbourCache end
```

No implementations exist yet, and nothing in the package calls these stubs. The natural consumers are:

- **2nd-order conservative regridding** (umbrella PR #93, branch `as/secondorder`) — needs neighbour cells to compute gradient stencils.
- **Non-conservative interpolators** (bilinear / nearest-neighbour) — need `findidx` to locate a point and then `neighbours` for a stencil.

This design instantiates `neighbours(grid, idx)` for the **cubed sphere** (the only one in this repo today, ClimaCore-backed) and the **longitude-latitude grid** (`RegularGrid` with `GO.Spherical()` manifold). `findidx`, `dual_neighbours`, and `AbstractNeighbourCache` are out of scope for this round.

## Contract

```julia
neighbours(grid, idx::Integer) -> AbstractArray{<:Integer}
```

- Returns linear global indices (not `CartesianIndex`), matching the regridder's sparse-matrix row/column convention and the `(face_idx-1)*ne² + face_local_idx` scheme already used by `IndexOffsetQuadtreeCursor`.
- Length is variable: 8 for typical interior cells, 7 at cubed-sphere cube corners, fewer at non-periodic lon-lat borders.
- The `AbstractArray` contract leaves room for callers (e.g. HEALPix-style preallocation) to specialize the return type with a fixed-size buffer + count. v1 returns `Vector{Int}` with `sizehint!(_, 8)`.
- No orientation transform is returned. Callers that need cubed-sphere edge orientation for gradient computation will need a separate accessor (deferred).

`has_optimized_neighbour_search` returns `true` for both new types.

## Architecture

Three additions in `src/trees/wrappers.jl`, one extension change.

### 1. `CubeFaceConnectivity` (new struct, internal)

A 6×4 static table populated at `treeify` time from ClimaCore's topology. For each `(face_idx, edge_id)` it stores `(neighbour_face::Int8, neighbour_edge::Int8, reversed::Bool)`.

Edge IDs match ClimaCore's element-local face numbering (verified by running ClimaCore on `ne=2`):

```
1 = south (j_min)
2 = east  (i_max)
3 = north (j_max)
4 = west  (i_min)
```

The `reversed` field is kept (rather than dropped as always-true) so the table is self-describing and populated by what ClimaCore actually returned. Cost: 24 bools.

### 2. `CubedSphereToplevelTree` gains a `connectivity::CubeFaceConnectivity` field

Existing struct in `src/trees/wrappers.jl`. The field is added directly (not via a type parameter) so dispatch isn't perturbed. The only producer is the ClimaCore extension, so updating its constructor call is the only ripple.

### 3. `LonLatConnectivityWrapper{T}` (new wrapper)

```julia
struct LonLatConnectivityWrapper{T}
    tree::T
    periodic_x::Bool
    pole_top_fold::Bool
    pole_bottom_fold::Bool
    nlon::Int
    nlat::Int
end
```

The constructor takes `tree` (typically a `TopDownQuadtreeCursor{<:RegularGrid}`), reads grid coordinates, and infers the three booleans. `nlon`/`nlat` are cached for the per-query modulo arithmetic.

**Inference rules** (tolerance `atol = 3.6e-4°`, ≈ `1e-6 * 360°`):

- `periodic_x = (x[end] - x[1]) ≈ 360°`
- `pole_top_fold = y[end] ≈ 90° && iseven(nlon)`
- `pole_bottom_fold = y[1] ≈ -90° && iseven(nlon)`
- For `Planar()` manifold: all three are forced false.
- For odd `nlon`, the fold is mathematically ill-defined; flag is forced false (cells at pole rows return fewer neighbours rather than erroring).

**Pass-through methods**: `STI.isspatialtree`, `STI.nchild`, `STI.getchild`, `STI.isleaf`, `STI.node_extent`, `STI.child_indices_extents`, `Trees.getgrid`. The wrapper is invisible to existing dual-DFS / regridder code; only `neighbours` does work.

## Cubed-sphere algorithm

Decode `idx → (face_idx, i, j)`:

```
face_idx       = ((idx-1) ÷ ne²) + 1
face_local_idx = ((idx-1) mod ne²) + 1
i              = mod1(face_local_idx, ne)
j              = ((face_local_idx-1) ÷ ne) + 1
```

Iterate the 8 CCW offsets `(di, dj) ∈ {S, SE, E, NE, N, NW, W, SW}`. For each:

```
i_new, j_new = i+di, j+dj
in_i = 1 ≤ i_new ≤ ne
in_j = 1 ≤ j_new ≤ ne

if in_i && in_j:                      # same face
    push linear(face_idx, i_new, j_new)
elif in_i ⊕ in_j:                     # exactly one out — single edge crossed
    edge_id = which_edge(i_new, j_new, ne)
    other_face, other_edge, reversed = connectivity[face_idx, edge_id]
    s       = (edge_id ∈ {1,3}) ? i_new : j_new   # along-edge coord ∈ 1..ne
    s_eff   = reversed ? (ne+1 - s) : s
    other_i, other_j = step_in_from_edge(other_edge, s_eff, ne)
    push linear(other_face, other_i, other_j)
else:                                 # both out — cube-corner diagonal — drop
    skip
```

**Cube-corner case falls out for free**. Corner cells `(1,1) (1,ne) (ne,1) (ne,ne)` are the only ones where both axes can simultaneously overflow, and only in *one* of the 8 offsets — the one pointing through the cube corner where 3 faces meet. That slot is dropped, giving exactly 7 neighbours at corners and 8 elsewhere. No per-corner table needed.

**`reversed` semantics**: `reversed=true` means parameterizations along the shared edge run in opposite directions, so `s_other = ne + 1 − s_source`. Verified empirically: ClimaCore's `Topology2D` returns `reversed=true` for *every* shared edge — so in practice the algorithm always swaps. The field stays in the table for honesty / future-proofing rather than for runtime branching.

**Result**: `Vector{Int}` of length 7 or 8. `sizehint!(_, 8)` on construction. **TODO**: replace per-call allocation with a preallocated buffer once `AbstractNeighbourCache` lands (deferred).

### Verified cube connectivity (concrete values for tests)

From running ClimaCore at `ne=2` (face axes: F1=+x, F2=+y, F3=+z, F4=−x, F5=−y, F6=−z):

```
table[1, S] = (F=6, edge=N, reversed=true)
table[1, E] = (F=2, edge=W, reversed=true)
table[1, N] = (F=3, edge=W, reversed=true)
table[1, W] = (F=5, edge=N, reversed=true)

table[5, S] = (F=4, edge=N, reversed=true)
table[5, E] = (F=6, edge=W, reversed=true)
table[5, N] = (F=1, edge=W, reversed=true)
table[5, W] = (F=3, edge=N, reversed=true)
```

These are pinned in the test suite (section "Testing strategy").

## Lon-lat algorithm

Decode `idx → (i, j)` using column-major `i` fast: `i = mod1(idx, nlon)`, `j = ((idx-1) ÷ nlon) + 1`.

For each of 8 CCW offsets:

```
i_new = i + di
j_new = j + dj
valid = true

# 1. handle x out-of-range first
if i_new < 1 || i_new > nlon
    if periodic_x:  i_new = mod1(i_new, nlon)
    else:           valid = false

# 2. handle y out-of-range (after x is normalised)
if valid && j_new < 1
    if pole_bottom_fold:  i_new = mod1(i_new + nlon÷2, nlon); j_new = 1
    else:                 valid = false
elif valid && j_new > nlat
    if pole_top_fold:     i_new = mod1(i_new + nlon÷2, nlon); j_new = nlat
    else:                 valid = false

if valid:  push!(result, i_new + (j_new-1)*nlon)
```

X is handled before Y so combined wrap+fold corner cases (e.g. `(0, nlat+1)` at the top-left corner of a global grid) resolve correctly: longitude wraps first, then the wrapped column is folded.

**Duplicate caveat**: at a pole fold, the same physical cell can be reached via two offsets only when `nlon` is pathologically small (e.g. `nlon = 2`). Documented in the docstring; not deduped in v1. For `nlon ≥ 4` the 8 offsets give 8 distinct cells at fold rows.

**Non-fold pole row** (e.g. odd `nlon`, or `y[end]` not at 90°): cells at `j = nlat` simply return fewer neighbours.

## ClimaCore extension changes

`ext/ConservativeRegriddingClimaCoreExt.jl` currently constructs `CubedSphereToplevelTree(quadtrees)` from a ClimaCore mesh. Two changes:

### `build_cube_connectivity(topology, ne) -> CubeFaceConnectivity`

The hard bit: `Topologies.opposing_face(topology, elem, face)` is **element-local** — `face` is one of the four element-local faces of a single spectral element, not a cube edge.

Verified by direct inspection: for `Topology2D` on the default cubed-sphere mesh,

- `topology.elemorder == CartesianIndices((ne, ne, 6))`, so global element index = `i + (j-1)*ne + (F-1)*ne²` (matches our `IndexOffsetQuadtreeCursor` scheme exactly).
- Element-local face IDs: 1=south, 2=east, 3=north, 4=west — same as our cube-edge IDs.

Algorithm:

```julia
function build_cube_connectivity(topology, ne)
    table = Array{Tuple{Int8,Int8,Bool}, 2}(undef, 4, 6)  # (edge, face)
    for F in 1:6, edge in 1:4
        # Pick one representative element on this edge of this face.
        # For non-corner: use ne÷2 along the edge.
        i, j = edge_representative(edge, ne)              # see helper
        elem = i + (j-1)*ne + (F-1)*ne²
        # The element-local face that lies on the cube edge IS edge_id (1..4).
        opelem, opface, reversed = Topologies.opposing_face(topology, elem, edge)
        # Decode opelem -> (F', i', j')
        F_prime         = ((opelem - 1) ÷ ne²) + 1
        face_local      = ((opelem - 1) %  ne²) + 1
        # The other_edge is exactly opface (verified mapping).
        table[edge, F]  = (Int8(F_prime), Int8(opface), reversed)
    end
    return CubeFaceConnectivity(table)
end
```

Helper `edge_representative(edge, ne)`:

```
edge=1 (south): (ne÷2 max 1, 1)
edge=2 (east):  (ne, ne÷2 max 1)
edge=3 (north): (ne÷2 max 1, ne)
edge=4 (west):  (1, ne÷2 max 1)
```

The element representative is *interior* to the edge (not at a corner) so that the queried element-local face is unambiguously the cube-edge crossing.

### Constructor call site

Update the existing `treeify(::Spherical, ::AbstractCubedSphere)` path to pass the populated table:

```julia
connectivity = build_cube_connectivity(topology, ne)
return CubedSphereToplevelTree(quadtrees, connectivity)
```

### Risk flag

The cube-edge ↔ element-local-face mapping is the only fragile spot. Drift between our convention and ClimaCore's would be silent — neighbours would resolve to wrong cells across edges without erroring. The verification test suite (next section) pins this down with concrete table-value assertions.

## Testing strategy

### `test/trees/neighbours_cubed_sphere.jl` (new)

1. Build `ne=2` and `ne=4` cubed spheres via the existing extension path.
2. **Pin the connectivity table**: assert `table[1, 1..4]` and `table[5, 1..4]` match the values in "Verified cube connectivity" above. Catches face-numbering drift early.
3. **Interior-of-face cells** (`ne ≥ 4`): assert `length(neighbours) == 8` and the 8 indices match pure `(i+di, j+dj)` arithmetic.
4. **Edge-of-face cells, non-corner** (`ne ≥ 3`): assert `length == 8`, exactly 3 indices fall outside the source face, and each cross-face index decodes to a cell on the expected `F'` whose `(i', j')` is consistent with `Topologies.opposing_face` queried directly.
5. **Corner cells** `(F, 1, 1)`, `(F, 1, ne)`, `(F, ne, 1)`, `(F, ne, ne)` for all `F`: assert `length == 7` exactly.
6. **Symmetry**: for a sample of source cells, `B ∈ neighbours(A) ⟹ A ∈ neighbours(B)`.

### `test/trees/neighbours_lonlat.jl` (new)

1. Global grid `range(-180, 180, 9)` × `range(-90, 90, 5)`: assert `periodic_x=true, pole_top_fold=true, pole_bottom_fold=true`.
2. Interior cell — 8 neighbours, exact indices.
3. `i=1`: 8 neighbours, includes `i=nlon` from wrap.
4. Top row `j=nlat`: 8 = 2 same-row + 3 below + 3 across-pole at `j=nlat`. Verify the three across-pole `i`s are `mod1.(i .+ (-1:1) .+ nlon÷2, nlon)`.
5. Top-left corner `(1, nlat)`: combined wrap + fold, 8 neighbours, no duplicates expected at this `nlon`.
6. **Non-global grid** `0:90` × `0:45`: all three flags false. Border cell at `(1, 1)` returns only the 3 valid in-bounds neighbours.
7. **Odd-nlon global grid** (`nlon=17`): `pole_top_fold` and `pole_bottom_fold` are forced false. Top-row cell returns 5 neighbours.
8. Symmetry round-trip.

### Scale tests (tagged, may be skipped in fast CI)

- **Cubed sphere `ne=64`** (24,576 cells). Random-sample 1,000 cells across all 6 faces. Assert `length ∈ {7, 8}`. Assert symmetry on the sample. `@belapsed` soft-assert `< 1 µs/query`.
- **Lon-lat 720×360** (259,200 cells, quarter-degree global). Random-sample 1,000 cells. Assert `length == 8` for all sampled cells (interior + edge + pole-row, since this grid is global with even nlon). Assert symmetry on the sample. Same `@belapsed` soft-assert.

These catch accidental allocations or `Vector{Int}` growth becoming a problem before the preallocated-buffer path lands.

## File-by-file change list

**New:**

- `src/trees/neighbours.jl` — `CubeFaceConnectivity` struct, `LonLatConnectivityWrapper{T}` struct + constructor with the inference logic, two `neighbours(...)` methods, decoder helpers. Included from `Trees.jl` after `wrappers.jl`.
- `test/trees/neighbours_cubed_sphere.jl`, `test/trees/neighbours_lonlat.jl` — the test sets above.

**Modified:**

- `src/trees/wrappers.jl` — `CubedSphereToplevelTree` gains `connectivity::CubeFaceConnectivity` field; constructor signature updated.
- `src/trees/Trees.jl` — `include("neighbours.jl")`, export `LonLatConnectivityWrapper`. `CubeFaceConnectivity` stays unexported (internal).
- `ext/ConservativeRegriddingClimaCoreExt.jl` — add `build_cube_connectivity(topology, ne)`; pass the populated table to `CubedSphereToplevelTree(...)`.
- `test/runtests.jl` — `include` the two new test files.

`src/trees/neighbours_interface.jl` is **not** modified — the existing stubs remain the interface; we provide implementations.

## Deferred / out of scope for this round

- `findidx(grid, pos)::idx` for both grids.
- `dual_neighbours(grid, pos, neighbours)`.
- `AbstractNeighbourCache` and the preallocated-buffer return path.
- Cubed-sphere edge orientation accessor (needed for 2nd-order gradient computation across cube edges).
- Neighbours for HEALPix, Oceananigans `TripolarGrid`, and SpeedyWeather grids — each gets its own design once the consumer pattern is clearer.
- Non-`Topology2D` ClimaCore topologies (e.g. SFC-reordered). The existing extension already branches for these via `Reorderer2D`; the connectivity-build path will need an analogous branch.
