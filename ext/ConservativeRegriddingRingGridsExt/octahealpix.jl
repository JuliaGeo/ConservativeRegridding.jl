#=
## OctaHEALPix closed forms

The reduced OctaHEALPix grid (4 octahedral faces, pole to pole). Points-per-ring
varies, so the whole grid isn't curvilinear — but each face is: a quadrant is an
`nside × nside` `(r, c)` matrix (the 45°-rotated diamond), which the
`OctaHEALPixRingGrid` provider (providers.jl) exposes as a lattice face over the
forms below; any `nlat_half` works (the `(r,c) ↔ ring` conversions are closed-form).

`nside == nlat_half`. Geometry is OctaHEALPix's closed form (per-pixel,
allocation-free, reproduces `RingGrids.get_vertices` exactly).

Each pixel is a diamond whose corners sit on the latitude "rails" of neighbouring
rings — N corner on ring j-1, E/W on ring j, S on ring j+1. A rail at level
v = 0..2nside (0 = N pole, 2nside = S pole) has latitude from z = 1 - (v/nside)²
(Górski et al. 2005 eq. 4 with 3N²→N²); per-ring longitudes are equidistant.
=#

# Index spaces: ring-order ↔ (j, q, iq) ↔ (r, c) matrix.

# cells per quadrant on ring j (the ring holds 4× this many points)
_octa_cells_per_quadrant(j, nside) = min(j, 2nside - j)

# ring index of ring-order point `ij` (closed-form inverse of the cumulative ring counts)
function _octa_whichring(ij, nside)
    if ij <= 2nside * (nside + 1)              # northern hemisphere incl. equator
        return ceil(Int, (-1 + sqrt(1 + 2ij)) / 2)
    else                                       # south: mirror the count from the south pole
        s = ceil(Int, (-1 + sqrt(1 + 2 * (4nside^2 + 1 - ij))) / 2)
        return 2nside - s
    end
end

# first ring-order index of ring j
_octa_ring_start(j, nside) =
    j <= nside ? 2j * (j - 1) + 1 : 4nside^2 - 2(2nside - j) * (2nside - j + 1) + 1

# ring-order index → (ring j, quadrant q, in-quadrant index iq), all 1-based
function _octa_ring_jqi(ij, nside)
    j    = _octa_whichring(ij, nside)
    i0   = ij - _octa_ring_start(j, nside)             # 0-based within the ring
    nlon = 4 * _octa_cells_per_quadrant(j, nside)
    q    = mod(4i0 ÷ nlon, 4)                           # 0-based quadrant
    iq   = i0 - q * (nlon ÷ 4)                          # 0-based within the quadrant
    return j, q + 1, iq + 1
end

# (ring j, quadrant q, in-quadrant iq) → ring-order index (inverse of `_octa_ring_jqi`)
function _octa_jqi_ring(j, q, iq, nside)
    nlon = 4 * _octa_cells_per_quadrant(j, nside)
    i = iq + (q - 1) * (nlon ÷ 4)                       # 1-based index within the ring
    return _octa_ring_start(j, nside) + i - 1
end

# Per-face `(r, c)` matrix (1-based) ↔ (ring j, in-quadrant iq). The cell at matrix
# `(r, c)` sits on ring `j = r + c - 1`; `iq` runs out from the diamond's poleward tip.
_octa_rc_jqi(r, c, nside) = (r + c - 1, min(nside - r, c - 1) + 1)

# inverse: (ring j, in-quadrant iq) → matrix (r, c)
function _octa_jqi_rc(j, iq, nside)
    return j <= nside ? (j - iq + 1, iq) : (nside - iq + 1, j - nside + iq)
end

# Closed-form cell geometry: (j, q, iq) → corners as (lon°, lat°).

# latitude (°) of corner-rail level v = 0..2nside; ring centres share these rails.
# z = 1 - (v/nside)², lat = 90° - acos(z)  (Górski et al. 2005 eq. 4 with 3N²→N²)
function _octa_lat(v, nside)
    v == 0      && return 90.0
    v == 2nside && return -90.0
    return v <= nside ? 90 - acosd(1 - (v / nside)^2) :
                      -(90 - acosd(1 - ((2nside - v) / nside)^2))
end

# Corners (named E/S/W/N) of a cell on a northern or equatorial ring j ≤ nside.
# E/W sit on ring j's rail, N on ring j-1's, S on ring j+1's.
function _octa_north_corners(nside, j, q, iq)
    mj   = _octa_cells_per_quadrant(j, nside)
    clon = 90 * (q - 1) + (2iq - 1) * 45 / mj           # cell-centre longitude
    latj = _octa_lat(j, nside)
    E = (mod(clon + 45 / mj, 360.0), latj)              # centre ± half a cell in longitude
    W = (mod(clon - 45 / mj, 360.0), latj)
    N = j == 1 ? (clon, 90.0) :
        (mod(90 * (q - 1) + (iq - 1) * 90 / _octa_cells_per_quadrant(j - 1, nside), 360.0),
         _octa_lat(j - 1, nside))
    # S sits on ring j+1: equator-ward (more cells, same iq) except across the
    # equator, where j+1 is pole-ward (fewer cells, iq-1). The final ring's S
    # corner is the south pole (only at nside == 1).
    if (j + 1) == 2nside
        S = (clon, -90.0)
    else
        kS = (j + 1) <= nside ? iq : iq - 1
        S = (mod(90 * (q - 1) + kS * 90 / _octa_cells_per_quadrant(j + 1, nside), 360.0),
             _octa_lat(j + 1, nside))
    end
    return (; E, S, W, N)
end

# Corners for any ring j. The south hemisphere is the matching north ring mirrored
# across the equator: longitudes kept, latitudes negated, N/S corners swapped.
function _octa_corners(nside, j, q, iq)
    j <= nside && return _octa_north_corners(nside, j, q, iq)
    n = _octa_north_corners(nside, 2nside - j, q, iq)
    mirror((lon, lat)) = (lon, -lat)
    return (E = mirror(n.E), S = mirror(n.N), W = mirror(n.W), N = mirror(n.S))
end

# Vertex (lon°, lat°) of lattice point (r, c) in quadrant q — the corner shared by
# the cells meeting there. It sits on rail v = (r-1) + (c-1); rail longitudes are
# equidistant, indexed by `k` (north `k = c-1`, south `k = c-v+nside-1`; the two
# agree at the equator rail v = nside). Poles are longitude-degenerate.
function _octa_vertex_lonlat(q, r, c, nside)
    v = (r - 1) + (c - 1)
    lat = _octa_lat(v, nside)
    (v == 0 || v == 2nside) && return (90.0 * (q - 1), lat)
    m, k = v <= nside ? (v, c - 1) : (2nside - v, c - v + nside - 1)
    return (mod(90.0 * (q - 1) + k * (90.0 / m), 360.0), lat)
end
