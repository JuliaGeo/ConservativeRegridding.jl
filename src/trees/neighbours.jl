#=
# Neighbours

This file provides concrete implementations of the [`neighbours`](@ref) interface
declared in `neighbours_interface.jl`, for two grid families:

- Cubed sphere (via [`CubedSphereToplevelTree`](@ref) — typically built by the
  ClimaCore extension).
- Longitude-latitude regular grids (via [`LonLatConnectivityWrapper`](@ref)).

`findidx`, `dual_neighbours`, and `AbstractNeighbourCache` are out of scope here.

The `CubeFaceConnectivity` struct lives in `wrappers.jl` so that
`CubedSphereToplevelTree` can carry it as a field without forward-declaration
gymnastics.
=#

import GeometryOps: SpatialTreeInterface as STI
import GeometryOpsCore as GOCore
import GeometryOps as GO

# 8 CCW offsets, starting at south and rotating counter-clockwise:
# S, SE, E, NE, N, NW, W, SW
const _NEIGHBOUR_OFFSETS = (
    (0, -1),   # S
    (1, -1),   # SE
    (1,  0),   # E
    (1,  1),   # NE
    (0,  1),   # N
    (-1, 1),   # NW
    (-1, 0),   # W
    (-1,-1),   # SW
)

#=
## LonLatConnectivityWrapper

A pass-through tree wrapper that augments a regular lon/lat
[`TopDownQuadtreeCursor`](@ref) (built on a [`RegularGrid`](@ref)) with the
metadata needed to compute `neighbours` correctly across periodic-x and
pole-fold boundaries.

The constructor `LonLatConnectivityWrapper(tree)` infers the three
boolean flags from the underlying grid's coordinate vectors:

- `periodic_x`        : `x[end] - x[1] ≈ 360°`
- `pole_top_fold`     : `y[end] ≈  90°` and `iseven(nlon)`
- `pole_bottom_fold`  : `y[1]   ≈ -90°` and `iseven(nlon)`

For `Planar` manifolds all three are forced false. The wrapper is invisible
to existing dual-DFS / regridder code; only `neighbours` makes use of it.
=#

"""
    LonLatConnectivityWrapper(tree)

A pass-through wrapper around a `TopDownQuadtreeCursor` (or similar) wrapping
a `RegularGrid` on a `Spherical()` manifold, that records whether the grid is
periodic in longitude and whether it folds across either pole. These flags are
used by [`neighbours`](@ref) to wrap and fold linear indices correctly at
domain boundaries.

The convenience constructor reads the grid's `x`/`y` coordinate vectors and the
manifold to infer the flags. Tolerance for the periodicity / pole tests is
`3.6e-4°`. For `Planar()` manifolds, all three flags are forced false. For odd
`nlon`, both fold flags are forced false (the fold is mathematically
ill-defined; cells at the pole row simply return fewer neighbours).
"""
struct LonLatConnectivityWrapper{T}
    tree::T
    periodic_x::Bool
    pole_top_fold::Bool
    pole_bottom_fold::Bool
    nlon::Int
    nlat::Int
end

function LonLatConnectivityWrapper(tree)
    grid = getgrid(tree)
    grid isa RegularGrid || throw(ArgumentError(
        "LonLatConnectivityWrapper expects a RegularGrid; got $(typeof(grid))"
    ))
    x = grid.x
    y = grid.y
    nlon = length(x) - 1
    nlat = length(y) - 1
    manifold = grid.manifold

    atol = 3.6e-4 # ≈ 1e-6 * 360°
    if manifold isa GO.Planar
        periodic_x = false
        pole_top_fold = false
        pole_bottom_fold = false
    else
        periodic_x = isapprox(x[end] - x[1], 360.0; atol)
        # Folds only make sense when nlon is even; otherwise mod1(i + nlon÷2, nlon)
        # does not pair distinct cells across the pole.
        pole_top_fold = iseven(nlon) && isapprox(y[end],  90.0; atol)
        pole_bottom_fold = iseven(nlon) && isapprox(y[1],  -90.0; atol)
    end

    return LonLatConnectivityWrapper(tree, periodic_x, pole_top_fold, pole_bottom_fold, nlon, nlat)
end

# Pass-through STI methods so the wrapper is invisible to dual-DFS / regridder.
STI.isspatialtree(::Type{<: LonLatConnectivityWrapper}) = true
STI.nchild(w::LonLatConnectivityWrapper) = STI.nchild(w.tree)
STI.getchild(w::LonLatConnectivityWrapper, i::Int) = STI.getchild(w.tree, i)
STI.isleaf(w::LonLatConnectivityWrapper) = STI.isleaf(w.tree)
STI.node_extent(w::LonLatConnectivityWrapper) = STI.node_extent(w.tree)
STI.child_indices_extents(w::LonLatConnectivityWrapper) = STI.child_indices_extents(w.tree)

# Pass-through cell-access methods so callers (build_weights, areas, etc.)
# can use a wrapped tree the same way as the underlying one.
getcell(w::LonLatConnectivityWrapper) = getcell(w.tree)
getcell(w::LonLatConnectivityWrapper, args...) = getcell(w.tree, args...)
ncells(w::LonLatConnectivityWrapper) = ncells(w.tree)
ncells(w::LonLatConnectivityWrapper, dim::Int) = ncells(w.tree, dim)
cell_range_extent(w::LonLatConnectivityWrapper, args...) = cell_range_extent(w.tree, args...)

getgrid(w::LonLatConnectivityWrapper) = getgrid(w.tree)

#=
## Cubed-sphere neighbours

Decode the global index to `(face, i, j)`, walk the 8 CCW offsets, and resolve
edge crossings via the connectivity table. Cube-corner cells naturally drop one
neighbour (the diagonal pointing through the 3-face cube corner is the only
offset that overflows in *both* axes simultaneously) — so corners return 7
neighbours and other cells 8, with no special case in code.
=#

# Which cube edge a (i_new, j_new) overflow points to. Returns 1..4 matching
# the same numbering used by `CubeFaceConnectivity`:
# 1 = south (j_new < 1), 2 = east (i_new > ne),
# 3 = north (j_new > ne), 4 = west (i_new < 1).
@inline function _which_edge(i_new::Int, j_new::Int, ne::Int)
    if j_new < 1
        return 1
    elseif i_new > ne
        return 2
    elseif j_new > ne
        return 3
    else  # i_new < 1
        return 4
    end
end

# Step one cell into the destination face from the given (other-face) edge,
# at the along-edge coordinate `s` (∈ 1..ne).
@inline function _step_in_from_edge(other_edge::Integer, s::Int, ne::Int)
    if other_edge == 1        # south: j = 1, i = s
        return s, 1
    elseif other_edge == 2    # east:  i = ne, j = s
        return ne, s
    elseif other_edge == 3    # north: j = ne, i = s
        return s, ne
    else                      # west:  i = 1, j = s
        return 1, s
    end
end

@inline function _cube_linear(face::Integer, i::Integer, j::Integer, ne::Integer)
    return Int(i) + (Int(j) - 1) * Int(ne) + (Int(face) - 1) * Int(ne)^2
end

"""
    neighbours(tree::CubedSphereToplevelTree, idx::Integer) -> Vector{Int}

Return the linear global indices of cells that share an edge or corner with the
cell at `idx` on a cubed sphere. Length is 8 in the interior of a face or
across an edge, and 7 at the eight cube corners (where three faces meet).

Indices follow the same `face_local + (face-1)*ne²` scheme used by
`IndexOffsetQuadtreeCursor`, so the returned indices match the regridder's
sparse-matrix row/column convention.
"""
function neighbours(tree::CubedSphereToplevelTree, idx::Integer)
    # TODO: replace per-call allocation with preallocated buffer when AbstractNeighbourCache lands
    conn = tree.connectivity
    ne = conn.ne
    table = conn.table

    ne2 = ne * ne
    face_idx = ((Int(idx) - 1) ÷ ne2) + 1
    face_local_idx = ((Int(idx) - 1) % ne2) + 1
    i = mod1(face_local_idx, ne)
    j = ((face_local_idx - 1) ÷ ne) + 1

    result = Int[]
    sizehint!(result, 8)

    for (di, dj) in _NEIGHBOUR_OFFSETS
        i_new = i + di
        j_new = j + dj
        in_i = 1 <= i_new <= ne
        in_j = 1 <= j_new <= ne

        if in_i && in_j
            push!(result, _cube_linear(face_idx, i_new, j_new, ne))
        elseif in_i ⊻ in_j
            edge_id = _which_edge(i_new, j_new, ne)
            other_face, other_edge, reversed = table[edge_id, face_idx]
            # along-edge coord on the source side, ∈ 1..ne
            s = (edge_id == 1 || edge_id == 3) ? i_new : j_new
            s_eff = reversed ? (ne + 1 - s) : s
            other_i, other_j = _step_in_from_edge(other_edge, s_eff, ne)
            push!(result, _cube_linear(other_face, other_i, other_j, ne))
        else
            # Both axes overflow simultaneously — diagonal through a cube corner
            # where 3 faces meet. Skip; this is the corner-with-7-neighbours case.
            continue
        end
    end

    return result
end

has_optimized_neighbour_search(::CubedSphereToplevelTree) = true

#=
## Lon-lat neighbours

Decode the global linear index assuming column-major i-fast layout
(`i = mod1(idx, nlon)`, `j = ((idx-1) ÷ nlon) + 1`) — matching how
`RegularGrid` cells are linearised by `cartesian_to_linear_idx`.

X is normalised before Y so combined wrap + fold corner cases (e.g.
`(0, nlat+1)` at the top-left of a global grid) resolve correctly: longitude
wraps first, then the wrapped column folds.
=#

"""
    neighbours(w::LonLatConnectivityWrapper, idx::Integer) -> Vector{Int}

Return the linear global indices of cells adjacent to the cell at `idx` on a
regular lon/lat grid. The exact set depends on the wrapper's `periodic_x`,
`pole_top_fold`, `pole_bottom_fold` flags:

- Interior cells return 8 neighbours.
- Periodic-x boundary cells wrap longitudinally.
- Pole-fold rows route across-pole neighbours via `mod1(i + nlon÷2, nlon)`.
- Non-periodic / non-fold borders return fewer neighbours (no padding).

When both `nlon ≥ 4` and the relevant flags are set, the 8 offsets always
resolve to 8 distinct cells. For pathologically small `nlon` (e.g. `nlon = 2`)
two offsets can collide at a fold row; duplicates are not removed in v1.
"""
function neighbours(w::LonLatConnectivityWrapper, idx::Integer)
    # TODO: replace per-call allocation with preallocated buffer when AbstractNeighbourCache lands
    nlon = w.nlon
    nlat = w.nlat
    periodic_x = w.periodic_x
    pole_top_fold = w.pole_top_fold
    pole_bottom_fold = w.pole_bottom_fold

    i = mod1(Int(idx), nlon)
    j = ((Int(idx) - 1) ÷ nlon) + 1

    result = Int[]
    sizehint!(result, 8)

    half = nlon ÷ 2

    for (di, dj) in _NEIGHBOUR_OFFSETS
        i_new = i + di
        j_new = j + dj
        valid = true

        # Handle x out-of-range first so combined wrap+fold corners resolve.
        if i_new < 1 || i_new > nlon
            if periodic_x
                i_new = mod1(i_new, nlon)
            else
                valid = false
            end
        end

        # Then handle y out-of-range, after x has been normalised.
        if valid && j_new < 1
            if pole_bottom_fold
                i_new = mod1(i_new + half, nlon)
                j_new = 1
            else
                valid = false
            end
        elseif valid && j_new > nlat
            if pole_top_fold
                i_new = mod1(i_new + half, nlon)
                j_new = nlat
            else
                valid = false
            end
        end

        if valid
            push!(result, i_new + (j_new - 1) * nlon)
        end
    end

    return result
end

has_optimized_neighbour_search(::LonLatConnectivityWrapper) = true
