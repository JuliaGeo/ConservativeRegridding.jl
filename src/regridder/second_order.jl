#=
# Second-order conservative regridding

This file contains the entire implementation of the second-order conservative
regridding algorithm. It implements ESMF's `CONSERVE_2ND` algorithm: a Taylor
reconstruction inside each source cell, with a gradient computed via discrete
Green's theorem on a polygon formed by the centroids of node-neighbour cells.

The first-order infrastructure (the `Regridder` struct, `RegriddingAlgorithm`
abstract type, `FirstOrderConservative`, and the dispatch entry point
`build_weights`) lives in `regridder.jl`. The 1st-order path of `build_weights`
calls `intersection_areas` from `intersection_areas.jl`.

Reference: `src/Infrastructure/Mesh/src/Regridding/ESMCI_Conserve2ndInterp.C` in
the [esmf-org/esmf](https://github.com/esmf-org/esmf) repository.

The algorithm is documented in `docs/plans/2026-04-26-second-order-design.md`.
=#

import StaticArrays
import LinearAlgebra
import GeometryOps as GO
import GeometryOpsCore as GOCore
import GeoInterface as GI
import SparseArrays
import ChunkSplitters
import StableTasks
using GeometryOps: SpatialTreeInterface as STI

# ---------------------------------------------------------------------------
# Algorithm types
# ---------------------------------------------------------------------------

"""
    abstract type CentroidAlgorithm

How the source-cell centroid (the Taylor expansion point) and neighbour-cell
centroids (the gradient stencil) are computed.

Concrete subtypes:
- [`ESMFLike`](@ref) (default) — vertex-mean projected to the sphere for
  neighbours; area-weighted from supermesh cells for the source-cell centroid.
- [`GeometricCentroid`](@ref) — `GO.centroid_and_area` for both. More accurate
  for curved cells, slightly slower.
"""
abstract type CentroidAlgorithm end

"""
    ESMFLike()

ESMF-style centroid: source-cell centroid is the area-weighted sum of
supermesh-cell vertex-mean centroids; neighbour centroids are the vertex-mean
of the neighbour polygon (projected to the sphere on `Spherical()`).
"""
struct ESMFLike <: CentroidAlgorithm end

"""
    GeometricCentroid()

Use `GO.centroid_and_area(manifold, polygon)` to compute every centroid.
"""
struct GeometricCentroid <: CentroidAlgorithm end

"""
    SecondOrderConservative([centroid_algorithm = ESMFLike()])

Second-order conservative regridding: preserves the integral and the
linear gradient inside each source cell. Construct a regridder with

```julia
Regridder(dst, src; algorithm = SecondOrderConservative())
```

`Base.transpose` is intentionally *not* defined for second-order regridders
(`supports_transpose` returns `false`); attempting to transpose raises a
`MethodError`.
"""
struct SecondOrderConservative{C <: CentroidAlgorithm} <: RegriddingAlgorithm
    centroid_algorithm::C
end
SecondOrderConservative() = SecondOrderConservative(ESMFLike())

# `supports_transpose(::SecondOrderConservative)` falls through to the
# `supports_transpose(::RegriddingAlgorithm) = false` default in regridder.jl.

# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------

@inline _spatial_dim(::Spherical) = 3
@inline _spatial_dim(::Planar) = 2

# Convert a GeoInterface point to an SVector in the manifold's native space.
@inline function _point_to_svec(::Spherical, ::Type{T}, p) where {T}
    if p isa GO.UnitSpherical.UnitSphericalPoint
        # UnitSphericalPoint is a FieldVector{3}; supports indexing.
        return StaticArrays.SVector{3, T}(T(p[1]), T(p[2]), T(p[3]))
    else
        u = GO.UnitSphereFromGeographic()((GI.x(p), GI.y(p)))
        return StaticArrays.SVector{3, T}(T(u[1]), T(u[2]), T(u[3]))
    end
end

@inline function _point_to_svec(::Planar, ::Type{T}, p) where {T}
    return StaticArrays.SVector{2, T}(T(GI.x(p)), T(GI.y(p)))
end

# Vertices of a polygon's exterior ring, dropping the duplicated closing point.
function _polygon_vertices(manifold::M, poly, ::Type{T}) where {M, T}
    ring = GI.getexterior(poly)
    npts = GI.npoint(ring)
    n = npts - 1  # closed ring duplicates first point at the end
    SV = StaticArrays.SVector{_spatial_dim(manifold), T}
    vs = Vector{SV}(undef, n)
    for i in 1:n
        vs[i] = _point_to_svec(manifold, T, GI.getpoint(ring, i))
    end
    return vs
end

# Single-pass, allocation-free, dedup-aware vertex mean of a polygon's
# exterior ring. Skips a vertex if it is within tolerance of the previous
# kept vertex; also compares the final kept vertex against the first to
# guard the wrap-around. Matches ESMF's MONOPOLE behaviour at the poles
# of a global lat/lon grid: the two top corners of a polar cap collapse
# to a single UnitSphericalPoint, so the cell's centroid is computed as
# if it were a triangle.
@inline _vmd_tol(::Spherical, ::Type{T}) where {T} = T(1e-12)
@inline _vmd_tol(::Planar,    ::Type{T}) where {T} = eps(T)  # scaled per pair

@inline function _vmd_close(::Spherical, p, q, tol)
    # UnitSphericalPoints are unit-norm, so an absolute chord tolerance is fine.
    d2 = (p[1] - q[1])^2 + (p[2] - q[2])^2 + (p[3] - q[3])^2
    return d2 <= tol * tol
end
@inline function _vmd_close(::Planar, p, q, tol)
    # Scale by point magnitude so Float32 rings at large coordinates still dedup.
    s2 = max(p[1]^2 + p[2]^2, q[1]^2 + q[2]^2, one(eltype(p)))
    d2 = (p[1] - q[1])^2 + (p[2] - q[2])^2
    return d2 <= (tol * tol) * s2
end

@inline _vmd_finalize(::Planar,    s, n) = s ./ n
@inline _vmd_finalize(::Spherical, s, n) = LinearAlgebra.normalize(s ./ n)

function _vertex_mean_dedup(manifold::M, poly, ::Type{T}) where {M <: Manifold, T <: AbstractFloat}
    ring = GI.getexterior(poly)
    npts = GI.npoint(ring)
    n_unique_max = npts - 1
    SV = StaticArrays.SVector{_spatial_dim(manifold), T}
    tol = _vmd_tol(manifold, T)

    first_v = _point_to_svec(manifold, T, GI.getpoint(ring, 1))
    s::SV = first_v
    prev::SV = first_v
    n = 1

    @inbounds for i in 2:n_unique_max
        v = _point_to_svec(manifold, T, GI.getpoint(ring, i))
        _vmd_close(manifold, v, prev, tol) && continue
        s = s + v
        prev = v
        n += 1
    end
    if n > 1 && _vmd_close(manifold, prev, first_v, tol)
        s = s - prev
        n -= 1
    end
    return _vmd_finalize(manifold, s, T(n))::SV
end

# Multipart-aware area-weighted vertex-mean centroid (ESMFLike) of an
# intersection-polygon list. Returns (centroid, total_area).
function _vertex_mean_centroid(manifold::Spherical, ::Type{T}, polys) where {T}
    accum = zero(StaticArrays.SVector{3, T})
    total = zero(T)
    for poly in polys
        a = T(GO.area(manifold, poly))
        a > 0 || continue
        accum = accum + a * _vertex_mean_dedup(manifold, poly, T)
        total += a
    end
    total > 0 || return zero(StaticArrays.SVector{3, T}), zero(T)
    return LinearAlgebra.normalize(accum), total
end

function _vertex_mean_centroid(manifold::Planar, ::Type{T}, polys) where {T}
    accum = zero(StaticArrays.SVector{2, T})
    total = zero(T)
    for poly in polys
        a = T(GO.area(manifold, poly))
        a > 0 || continue
        accum = accum + a * _vertex_mean_dedup(manifold, poly, T)
        total += a
    end
    total > 0 || return zero(StaticArrays.SVector{2, T}), zero(T)
    return accum ./ total, total
end

# Multipart-aware area-weighted geometric centroid (GeometricCentroid).
function _geometric_centroid(manifold::M, ::Type{T}, polys) where {M, T}
    SV = StaticArrays.SVector{_spatial_dim(manifold), T}
    accum = zero(SV)
    total = zero(T)
    for poly in polys
        c, a = GO.centroid_and_area(manifold, poly)
        a > 0 || continue
        accum = accum + T(a) * _point_to_svec(manifold, T, c)
        total += T(a)
    end
    total > 0 || return zero(SV), zero(T)
    if manifold isa Spherical
        return LinearAlgebra.normalize(accum), total
    else
        return accum ./ total, total
    end
end

# Intersection-polygon list for a (src, dst) pair. Algorithm-dispatched on
# manifold. Always returns a `Vector{<:GI.Polygon}` so the downstream multipart
# loop is uniform — Spherical's `ConvexConvexSutherlandHodgman` returns a
# single polygon (or an empty polygon), which we normalise here.
function _intersection_polys(manifold::Planar, p_src, p_dst)
    return GO.intersection(GO.FosterHormannClipping(manifold), p_src, p_dst;
                            target = GI.PolygonTrait())
end

function _intersection_polys(manifold::Spherical, p_src, p_dst)
    poly = GO.intersection(GO.ConvexConvexSutherlandHodgman(manifold), p_src, p_dst;
                            target = GI.PolygonTrait())
    # An empty intersection comes back as a polygon with 0 exterior points.
    n = GI.npoint(GI.getexterior(poly))
    return n == 0 ? typeof(poly)[] : [poly]
end

# Spherical polygon area via L'Huilier's theorem (anchored at the first vertex).
function _spherical_polygon_area(pts::AbstractVector{<:StaticArrays.SVector{3, T}}) where {T}
    n = length(pts); n < 3 && return zero(T)
    p1 = pts[1]
    a = zero(T)
    for i in 2:n-1
        a += _spherical_triangle_area(p1, pts[i], pts[i+1])
    end
    return a
end

function _spherical_triangle_area(p1::StaticArrays.SVector{3, T}, p2, p3) where {T}
    # L'Huilier's theorem; arc-side angles
    sa = acos(clamp(LinearAlgebra.dot(p2, p3), -one(T), one(T)))
    sb = acos(clamp(LinearAlgebra.dot(p1, p3), -one(T), one(T)))
    sc = acos(clamp(LinearAlgebra.dot(p1, p2), -one(T), one(T)))
    s  = (sa + sb + sc) / 2
    e_arg = tan(s/2) * tan((s - sa)/2) * tan((s - sb)/2) * tan((s - sc)/2)
    e_arg = max(e_arg, zero(T))
    return T(4) * atan(sqrt(e_arg))
end

# Planar polygon area via shoelace (assumes CCW orientation; we take |·|).
function _planar_polygon_area(pts::AbstractVector{<:StaticArrays.SVector{2, T}}) where {T}
    n = length(pts); n < 3 && return zero(T)
    a = zero(T)
    @inbounds for i in 1:n
        j = i == n ? 1 : i + 1
        a += pts[i][1] * pts[j][2] - pts[j][1] * pts[i][2]
    end
    return abs(a) / 2
end

@inline _polygon_area_from_pts(::Spherical, pts) = _spherical_polygon_area(pts)
@inline _polygon_area_from_pts(::Planar, pts)    = _planar_polygon_area(pts)

# Point-in-polygon test on the gradient-stencil polygon (vertices c_m).
# Used to detect a degenerate stencil (r_n outside the centroid hull).
function _point_in_polygon(::Planar, r_n::StaticArrays.SVector{2, T}, pts) where {T}
    n = length(pts); n < 3 && return false
    inside = false
    @inbounds for i in 1:n
        j = i == 1 ? n : i - 1
        xi, yi = pts[i][1], pts[i][2]
        xj, yj = pts[j][1], pts[j][2]
        # Ray-cast: skip horizontal edges (yj == yi) via the straddle check.
        if ((yi > r_n[2]) != (yj > r_n[2])) &&
           (r_n[1] < (xj - xi) * (r_n[2] - yi) / (yj - yi) + xi)
            inside = !inside
        end
    end
    return inside
end

function _point_in_polygon(::Spherical, r_n::StaticArrays.SVector{3, T}, pts) where {T}
    # Spherical turn-sign test. `pts` is the CCW-sorted centroid hull around
    # `r_n`; the point is inside iff it lies on the "interior" side of every
    # great-circle edge. Uses GeometryOps' `robust_cross_product`, which falls
    # back to extended precision when the two endpoints are nearly identical
    # or antipodal (where a plain cross product loses orthogonality). Tolerance
    # `16 * eps` matches `spherical_orient`.
    n = length(pts); n < 3 && return false
    tol = T(16) * eps(T)
    @inbounds for i in 1:n
        j = i == n ? 1 : i + 1
        nrm = GO.UnitSpherical.robust_cross_product(pts[i], pts[j])
        d = LinearAlgebra.dot(nrm, r_n)
        d < -tol && return false
    end
    return true
end

# ---------------------------------------------------------------------------
# Phase 2 — per-pair (area, centroid) computation, parallelised
# ---------------------------------------------------------------------------

# Centroid of a single intersection-polygon list, dispatched on the centroid
# algorithm. Returns (centroid, total_area), or zero(centroid), zero(area)
# when the intersection has no positive-area component.
@inline function _sm_cell_centroid(::ESMFLike, manifold, ::Type{T}, polys) where {T}
    return _vertex_mean_centroid(manifold, T, polys)
end
@inline function _sm_cell_centroid(::GeometricCentroid, manifold, ::Type{T}, polys) where {T}
    return _geometric_centroid(manifold, T, polys)
end

function _compute_sm_cells_chunk(alg::SecondOrderConservative, manifold::M, dst_tree, src_tree,
                                  idxs::AbstractVector{Tuple{Int, Int}}, ::Type{T}) where {M, T}
    SV = StaticArrays.SVector{_spatial_dim(manifold), T}
    Idst = Int[]; Isrc = Int[]; As = T[]; Cs = SV[]
    sizehint!(Idst, length(idxs)); sizehint!(Isrc, length(idxs))
    sizehint!(As, length(idxs)); sizehint!(Cs, length(idxs))

    for (i_src, i_dst) in idxs
        p_src = Trees.getcell(src_tree, i_src)
        p_dst = Trees.getcell(dst_tree, i_dst)
        polys = _intersection_polys(manifold, p_src, p_dst)
        isempty(polys) && continue
        c, a = _sm_cell_centroid(alg.centroid_algorithm, manifold, T, polys)
        a > 0 || continue
        push!(Idst, i_dst); push!(Isrc, i_src)
        push!(As, a); push!(Cs, c)
    end
    return Idst, Isrc, As, Cs
end

function _compute_sm_cells(alg::SecondOrderConservative, manifold::M, threaded::True, dst_tree, src_tree,
                            candidates::AbstractVector{Tuple{Int, Int}}, ::Type{T}) where {M, T}
    npart = max(1, Threads.nthreads() * 4)
    parts = ChunkSplitters.chunks(candidates; n = npart)
    tasks = [
        StableTasks.@spawn _compute_sm_cells_chunk($alg, $manifold, $dst_tree, $src_tree, part, $T)
        for part in parts
    ]
    results = map(fetch, tasks)
    return (
        reduce(vcat, getindex.(results, 1)),
        reduce(vcat, getindex.(results, 2)),
        reduce(vcat, getindex.(results, 3)),
        reduce(vcat, getindex.(results, 4)),
    )
end

function _compute_sm_cells(alg::SecondOrderConservative, manifold::M, ::False, dst_tree, src_tree,
                            candidates::AbstractVector{Tuple{Int, Int}}, ::Type{T}) where {M, T}
    return _compute_sm_cells_chunk(alg, manifold, dst_tree, src_tree, candidates, T)
end

# ---------------------------------------------------------------------------
# Phase 3 — gradient + weight emission per source cell
# ---------------------------------------------------------------------------

# Source centroid r_n.
function _source_centroid(::SecondOrderConservative{ESMFLike}, manifold::Spherical, src_tree, n,
                          sm_areas, sm_cntrs, ::Type{T}) where {T}
    accum = zero(StaticArrays.SVector{3, T})
    total = zero(T)
    @inbounds for k in eachindex(sm_areas)
        accum = accum + sm_areas[k] * sm_cntrs[k]
        total += sm_areas[k]
    end
    if total > 0
        return LinearAlgebra.normalize(accum)
    else
        return _vertex_mean_dedup(manifold, Trees.getcell(src_tree, n), T)
    end
end

function _source_centroid(::SecondOrderConservative{ESMFLike}, manifold::Planar, src_tree, n,
                          sm_areas, sm_cntrs, ::Type{T}) where {T}
    accum = zero(StaticArrays.SVector{2, T})
    total = zero(T)
    @inbounds for k in eachindex(sm_areas)
        accum = accum + sm_areas[k] * sm_cntrs[k]
        total += sm_areas[k]
    end
    if total > 0
        return accum ./ total
    else
        return _vertex_mean_dedup(manifold, Trees.getcell(src_tree, n), T)
    end
end

function _source_centroid(::SecondOrderConservative{GeometricCentroid}, manifold, src_tree, n,
                          _sm_areas, _sm_cntrs, ::Type{T}) where {T}
    p = Trees.getcell(src_tree, n)
    c, _ = GO.centroid_and_area(manifold, p)
    return _point_to_svec(manifold, T, c)
end

# Neighbour centroid c_m.
function _neighbour_centroid(::SecondOrderConservative{ESMFLike}, manifold::M, poly_m, ::Type{T}) where {M <: Manifold, T}
    return _vertex_mean_dedup(manifold, poly_m, T)
end

function _neighbour_centroid(::SecondOrderConservative{GeometricCentroid}, manifold, poly_m, ::Type{T}) where {T}
    c, _ = GO.centroid_and_area(manifold, poly_m)
    return _point_to_svec(manifold, T, c)
end

# CCW angle of vector b relative to v1, in r_n's tangent plane.
@inline function _ccw_angle(::Spherical, v1, b, n_hat)
    cross = LinearAlgebra.cross(v1, b)
    return atan(LinearAlgebra.dot(cross, n_hat), LinearAlgebra.dot(v1, b))
end
@inline function _ccw_angle(::Planar, v1, b, _n_hat)
    return atan(v1[1] * b[2] - v1[2] * b[1], LinearAlgebra.dot(v1, b))
end

# Reorder (nbr_ids, c_nbrs) CCW around r_n. Returns true on success.
# Reference vector v1 is the offset of the largest-id neighbour with c_m ≠ r_n.
function _sort_ccw!(nbr_ids::AbstractVector{Int}, c_nbrs::AbstractVector{<:StaticArrays.SVector},
                    r_n, u_src_cntr, manifold, ::Type{T}) where {T}
    N = length(nbr_ids)
    ref = 0
    @inbounds for i in 1:N
        if !isapprox(c_nbrs[i], r_n; atol = T(1e-14))
            if ref == 0 || nbr_ids[i] > nbr_ids[ref]
                ref = i
            end
        end
    end
    ref == 0 && return false
    v1 = c_nbrs[ref] - r_n
    angles = Vector{Float64}(undef, N)
    @inbounds for i in 1:N
        v = c_nbrs[i] - r_n
        angles[i] = _ccw_angle(manifold, v1, v, u_src_cntr)
    end
    perm = sortperm(angles)
    permute!(nbr_ids, perm)
    permute!(c_nbrs, perm)
    return true
end

# ---------------------------------------------------------------------------
# Per-source-cell driver
# ---------------------------------------------------------------------------

const _MAX_NBRS = 150  # ESMF's MAX_NUM_NBRS

# Process a single source cell `n`: emit weight entries for every (sm_cell, src_or_nbr).
# Pushes triples to local_I, local_J, local_V.
function _process_source_cell!(local_I::Vector{Int}, local_J::Vector{Int}, local_V::Vector{T},
                                alg::SecondOrderConservative, manifold::M, src_tree, n::Int,
                                sm_dsts::AbstractVector{Int}, sm_areas::AbstractVector{T},
                                sm_cntrs::AbstractVector,
                                dst_areas::Vector{T}) where {T, M}
    isempty(sm_dsts) && return

    # 1. Source centroid r_n
    r_n = _source_centroid(alg, manifold, src_tree, n, sm_areas, sm_cntrs, T)
    nrm = LinearAlgebra.norm(r_n)
    u_src_cntr = nrm > 0 ? r_n ./ nrm : r_n  # unit-normalized; only meaningful on Spherical

    # 2. Neighbour list and centroids
    nbr_ids = Trees.neighbours(src_tree, n)
    Nnbr = length(nbr_ids)

    # 4(a)/(b)/(f) early fallback paths — emit 1st-order weights only.
    if Nnbr < 3 || Nnbr > _MAX_NBRS
        _emit_weights_zero_grad!(local_I, local_J, local_V, n, sm_dsts, sm_areas, dst_areas)
        return
    end

    SV = StaticArrays.SVector{_spatial_dim(manifold), T}
    c_nbrs = Vector{SV}(undef, Nnbr)
    @inbounds for i in 1:Nnbr
        poly_m = Trees.getcell(src_tree, nbr_ids[i])
        c_nbrs[i] = _neighbour_centroid(alg, manifold, poly_m, T)
    end
    nbr_ids = collect(nbr_ids)  # ensure mutability for permute!

    # 3. CCW sort around r_n
    if !_sort_ccw!(nbr_ids, c_nbrs, r_n, u_src_cntr, manifold, T)
        # Fallback (f): no valid reference neighbour
        _emit_weights_zero_grad!(local_I, local_J, local_V, n, sm_dsts, sm_areas, dst_areas)
        return
    end

    # 4(c) Point-in-polygon: r_n must lie inside the centroid hull
    if !_point_in_polygon(manifold, r_n, c_nbrs)
        _emit_weights_zero_grad!(local_I, local_J, local_V, n, sm_dsts, sm_areas, dst_areas)
        return
    end

    # 4(d) zero-area neighbour polygon
    nbr_poly_area = _polygon_area_from_pts(manifold, c_nbrs)
    if nbr_poly_area <= 0
        _emit_weights_zero_grad!(local_I, local_J, local_V, n, sm_dsts, sm_areas, dst_areas)
        return
    end

    # 5. Green's-theorem gradient weights
    grad_per_nbr = fill(zero(SV), Nnbr)
    src_grad = zero(SV)
    abort = false
    @inbounds for i in 1:Nnbr
        prev = i == 1 ? Nnbr : i - 1
        ci = c_nbrs[i]; cp = c_nbrs[prev]
        contrib = if manifold isa Spherical
            n_hat = LinearAlgebra.cross(ci, cp)
            mag = LinearAlgebra.norm(n_hat)
            d = LinearAlgebra.dot(ci, cp)
            if mag <= T(1e-14)
                if d < 0
                    # 4(e) antipodal — fall back
                    abort = true
                    break
                else
                    # parallel/coincident — skip this edge
                    continue
                end
            end
            arc_len = acos(clamp(d, -one(T), one(T)))
            (n_hat / mag) * arc_len
        else
            edge = ci - cp
            StaticArrays.SVector{2, T}(edge[2], -edge[1])
        end
        grad_per_nbr[i]    = grad_per_nbr[i]    + contrib
        grad_per_nbr[prev] = grad_per_nbr[prev] + contrib
        src_grad           = src_grad           + contrib
    end
    if abort
        _emit_weights_zero_grad!(local_I, local_J, local_V, n, sm_dsts, sm_areas, dst_areas)
        return
    end

    # Asymmetric scaling: ÷2A for per-neighbour, ÷A for src_grad.
    if manifold isa Spherical
        # Tangent projection at r_n: g_t = (u_src × g) × u_src
        @inbounds for i in 1:Nnbr
            g = grad_per_nbr[i]
            g = LinearAlgebra.cross(LinearAlgebra.cross(u_src_cntr, g), u_src_cntr)
            grad_per_nbr[i] = g / (2 * nbr_poly_area)
        end
        src_grad = LinearAlgebra.cross(LinearAlgebra.cross(u_src_cntr, src_grad), u_src_cntr)
        src_grad = src_grad / nbr_poly_area
    else
        @inbounds for i in 1:Nnbr
            grad_per_nbr[i] = grad_per_nbr[i] / (2 * nbr_poly_area)
        end
        src_grad = src_grad / nbr_poly_area
    end

    # 6. Weight emission per (n, k)
    @inbounds for k in eachindex(sm_dsts)
        diff = sm_cntrs[k] - r_n
        ratio = sm_areas[k] / dst_areas[sm_dsts[k]]
        # Round-off clamp (matches ESMF)
        if one(T) < ratio < one(T) + T(1e-10)
            ratio = one(T)
        end
        # Source-cell weight at column n
        push!(local_I, sm_dsts[k]); push!(local_J, n)
        push!(local_V, ratio - LinearAlgebra.dot(diff, src_grad) * ratio)
        # Neighbour weights at columns m
        for j in 1:Nnbr
            push!(local_I, sm_dsts[k]); push!(local_J, nbr_ids[j])
            push!(local_V, LinearAlgebra.dot(diff, grad_per_nbr[j]) * ratio)
        end
    end
    return
end

# Fallback emission: 1st-order weights only (zero gradient).
function _emit_weights_zero_grad!(local_I::Vector{Int}, local_J::Vector{Int}, local_V::Vector{T},
                                   n::Int, sm_dsts::AbstractVector{Int},
                                   sm_areas::AbstractVector{T}, dst_areas::Vector{T}) where {T}
    @inbounds for k in eachindex(sm_dsts)
        ratio = sm_areas[k] / dst_areas[sm_dsts[k]]
        if one(T) < ratio < one(T) + T(1e-10)
            ratio = one(T)
        end
        push!(local_I, sm_dsts[k]); push!(local_J, n); push!(local_V, ratio)
    end
    return
end

# ---------------------------------------------------------------------------
# Phase 3 driver — group by source cell, parallelise, assemble matrix
# ---------------------------------------------------------------------------

# CSR-style group: returns offsets (cumsum of counts) and a permutation of
# sm-cell indices such that `perm[offsets[n]-counts[n]+1 : offsets[n]]` lists
# all sm-cells whose src is `n`.
function _group_by_src(Isrc::Vector{Int}, n_src::Int)
    counts = zeros(Int, n_src)
    @inbounds for s in Isrc
        counts[s] += 1
    end
    offsets = cumsum(counts)
    perm = Vector{Int}(undef, length(Isrc))
    write_pos = offsets .- counts
    @inbounds for k in eachindex(Isrc)
        s = Isrc[k]
        write_pos[s] += 1
        perm[write_pos[s]] = k
    end
    return offsets, counts, perm
end

function _process_source_cells_chunk!(alg::SecondOrderConservative, manifold, src_tree,
                                       chunk::AbstractVector{Int},
                                       sm_dsts_perm::Vector{Int},
                                       sm_areas_perm::Vector{T},
                                       sm_cntrs_perm::Vector,
                                       offsets::Vector{Int}, counts::Vector{Int},
                                       dst_areas::Vector{T}) where {T}
    local_I = Int[]; local_J = Int[]; local_V = T[]
    @inbounds for n in chunk
        c = counts[n]
        c == 0 && continue
        rng = (offsets[n] - c + 1):offsets[n]
        _process_source_cell!(local_I, local_J, local_V, alg, manifold, src_tree, n,
                               view(sm_dsts_perm, rng), view(sm_areas_perm, rng),
                               view(sm_cntrs_perm, rng), dst_areas)
    end
    return local_I, local_J, local_V
end

# ---------------------------------------------------------------------------
# build_weights — entry point dispatched from regridder.jl
# ---------------------------------------------------------------------------

function build_weights(alg::SecondOrderConservative, manifold::M, threaded, dst_tree, src_tree;
                        intersection_operator = nothing,
                        T::Type{<:AbstractFloat} = DEFAULT_FLOATTYPE,
                        kwargs...) where {M <: Manifold}

    if !Trees.has_optimized_neighbour_search(src_tree)
        throw(ArgumentError(
            "SecondOrderConservative requires a source tree that implements `Trees.neighbours`. " *
            "Got $(typeof(src_tree)). Wrap your source grid in a `LonLatConnectivityWrapper` " *
            "or use a cubed-sphere `CubedSphereToplevelTree`."
        ))
    end

    n_dst = prod(Trees.ncells(dst_tree))
    n_src = prod(Trees.ncells(src_tree))

    # Phase 1 — candidates.
    predicate_f = M <: Spherical ? GO.UnitSpherical._intersects : Extents.intersects
    candidates = get_all_candidate_pairs(threaded, predicate_f, src_tree, dst_tree)

    # Phase 2 — per-pair (area, centroid).
    Idst, Isrc, As, Cs = _compute_sm_cells(alg, manifold, threaded, dst_tree, src_tree, candidates, T)

    # Cache dst_areas in T (needed by emission step). Use Trees-derived areas
    # so that this build_weights is independent of the caller's `areas` cache.
    dst_areas_T = convert(Vector{T}, areas(manifold, dst_tree))

    # Group by src cell.
    offsets, counts, perm = _group_by_src(Isrc, n_src)
    sm_dsts_perm  = Idst[perm]
    sm_areas_perm = As[perm]
    sm_cntrs_perm = Cs[perm]

    # Phase 3 — process source cells in parallel, assemble matrix.
    src_chunks = ChunkSplitters.chunks(1:n_src; n = max(1, Threads.nthreads() * 4))
    if threaded isa True
        tasks = [
            StableTasks.@spawn _process_source_cells_chunk!(
                $alg, $manifold, $src_tree, collect(part),
                $sm_dsts_perm, $sm_areas_perm, $sm_cntrs_perm,
                $offsets, $counts, $dst_areas_T)
            for part in src_chunks
        ]
        results = map(fetch, tasks)
        I = reduce(vcat, getindex.(results, 1))
        J = reduce(vcat, getindex.(results, 2))
        V = reduce(vcat, getindex.(results, 3))
    else
        I, J, V = _process_source_cells_chunk!(alg, manifold, src_tree, collect(1:n_src),
                                                sm_dsts_perm, sm_areas_perm, sm_cntrs_perm,
                                                offsets, counts, dst_areas_T)
    end

    # The 1st-order assembly stores intersection areas (then `regrid!` divides
    # by dst_areas). For 2nd-order, we want `regrid!` to give the same final
    # result, so we store W * dst_areas[I] in the matrix. That way
    # `regrid!`'s `W * src ./ dst_areas` reproduces the intended W * src.
    @inbounds for k in eachindex(I)
        V[k] *= dst_areas_T[I[k]]
    end

    return SparseArrays.sparse(I, J, V, n_dst, n_src, +)
end
