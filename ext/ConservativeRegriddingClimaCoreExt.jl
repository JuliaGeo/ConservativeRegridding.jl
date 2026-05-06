module ConservativeRegriddingClimaCoreExt

import ConservativeRegridding
using ConservativeRegridding: Trees

import GeometryOpsCore as GOCore
import GeometryOps as GO
import GeoInterface as GI
import SparseArrays
import Extents

using GeometryOps.UnitSpherical: UnitSphericalPoint
using LinearAlgebra: normalize

using ClimaCore:
    CommonSpaces, Fields, Spaces, RecursiveApply, Meshes, Quadratures, Topologies, DataLayouts, ClimaComms
import ClimaCore

"""
    coords_for_face(mesh::CubedSphereMesh, face_idx)::Matrix{UnitSphericalPoint}

Get the normalized coordinates of each element vertex for a face of a cubed sphere.

For example, if the cubed sphere has 8 elements in each dimension of a panel (face),
this function will return a matrix of 9x9 points, each with normalized coordinates.
"""
function coords_for_face(mesh::Meshes.AbstractCubedSphere, face_idx)
    ne = mesh.ne
    coords = [
        begin
            coord = Meshes._coordinates(mesh, ϕx, ϕy, face_idx)
            usp = normalize(UnitSphericalPoint(coord.x1, coord.x2, coord.x3))
            usp
        end
        for ϕx in LinRange(-1, 1, ne+1), ϕy in LinRange(-1, 1, ne+1)
    ]

    return coords
end

function Trees.treeify(manifold::GOCore.Spherical, topology::Topologies.Topology2D{<: ClimaComms.AbstractCommsContext, <: Meshes.AbstractCubedSphere})
    mesh = topology.mesh
    ne = mesh.ne
    # There are always 6 faces of a cubed sphere -
    # see the ClimaCore docs on AbstractCubedSphere
    # for an explanation on how they are connected.
    # TODO: set the border elements of each cubed sphere face
    # explicitly equal to each other, so that there are no numerical
    # inaccuracies.
    face_idxs = 1:6
    face_coords = map(i -> coords_for_face(mesh, i), face_idxs)
    # Determine whether the elements are ordered in a space-filling curve
    # or a regular grid.
    quadtrees = if topology.elemorder isa CartesianIndices # Matrix order filling
        # Create a quadtree for each face.
        map(face_idxs, face_coords) do face_idx, coords
            Trees.IndexOffsetQuadtreeCursor(
                Trees.CellBasedGrid(manifold, coords),
                (face_idx - 1) * ne^2
            )
        end
    elseif topology.elemorder isa Vector{CartesianIndex{3}} # Some sort of space filling curve
        lin2carts = map.(
            i -> CartesianIndex((i[1], i[2])),
            Iterators.partition(topology.elemorder, length(topology.elemorder) ÷ 6)
        )
        cart2lins = map(enumerate(lin2carts)) do (face_idx, face_indices)
            mat = Matrix{Int}(undef, ne, ne)
            for (i, elem) in enumerate(face_indices)
                mat[elem] = i + (face_idx - 1) * ne^2  # Global index for child_indices_extents
            end
            mat
        end

        map(face_idxs, lin2carts, cart2lins, face_coords) do face_idx, lin2cart, cart2lin, coords
            Trees.IndexLocalizerRewrapperTree(
                Trees.ReorderedTopDownQuadtreeCursor(
                    Trees.CellBasedGrid(
                        GO.Spherical(; radius = mesh.domain.radius),
                        coords
                    ),
                    Trees.Reorderer2D(cart2lin, lin2cart)
                ),
                (face_idx - 1) * ne^2
            )
        end
    else
        error("Unknown spacefillingcurve type: $(typeof(topology.spacefillingcurve))\nExpected a CartesianIndices or a Vector{CartesianIndex{3}}")
    end

    return Trees.CubedSphereToplevelTree(quadtrees)
end

Trees.treeify(manifold::GOCore.Spherical, space::ClimaCore.Spaces.AbstractSpectralElementSpace) = Trees.treeify(manifold, Spaces.topology(space))

GOCore.best_manifold(mesh::Meshes.AbstractCubedSphere) = GOCore.Spherical(; radius = mesh.domain.radius)
GOCore.best_manifold(topology::Topologies.Topology2D) = GOCore.best_manifold(topology.mesh)
GOCore.best_manifold(space::ClimaCore.Spaces.AbstractSpectralElementSpace) = GOCore.best_manifold(Spaces.topology(space))

GOCore.best_manifold(field::ClimaCore.Fields.Field) = GOCore.best_manifold(getfield(field, :space))
Trees.treeify(manifold::GOCore.Spherical, field::ClimaCore.Fields.Field) = Trees.treeify(manifold, getfield(field, :space))

## Node extraction helpers for spectral element fields
"""
    _flat_nodal_data(p::AbstractArray) → Vector

Extract a flat vector from a parent array in IJFH or VIJFH data layout.
Ordering: i varies fastest, then j, then element h.

Note we regrid only 2D Fields here, so we can drop the
F and V indices.
"""
function _flat_nodal_data(p::AbstractArray)
    if ndims(p) == 4      # IJFH: (I, J, F, H)
        return collect(vec(view(p, :, :, 1, :)))
    elseif ndims(p) == 5  # VIJFH: (I, J, F, V, H)
        return collect(vec(view(p, :, :, 1, 1, :)))
    else
        error("Unexpected data layout dimensionality: $(ndims(p))")
    end
end

"""
    se_node_positions(space) → Vector{UnitSphericalPoint}

Extract the lat/lon positions of all spectral element nodes as a flat vector of
`UnitSphericalPoint`s.  Ordering: all Nq² nodes of element 1 (i fastest),
then element 2, etc.
"""
function se_node_positions(space)
    coords = Fields.coordinate_field(space)
    lat_flat  = _flat_nodal_data(parent(Fields.field_values(coords.lat)))
    long_flat = _flat_nodal_data(parent(Fields.field_values(coords.long)))
    transform = GO.UnitSphereFromGeographic()
    return [transform((long_flat[k], lat_flat[k])) for k in eachindex(lat_flat)]
end

"""
    se_node_weights(space) → Vector{Float64}

Extract the Jacobian integration weights ``W_{e,i,j}`` for all SE nodes as a
flat vector.  Same ordering as [`se_node_positions`](@ref).
"""
function se_node_weights(space)
    wj = Spaces.weighted_jacobian(space)
    return _flat_nodal_data(parent(wj))
end

# ────────────────────────────────────────────────────────────────────────────
# Inverse element map for the equiangular cubed sphere
# ────────────────────────────────────────────────────────────────────────────

"""
    element_face_local_indices(topology, elem_idx) -> (face, ie, je)

Return the cubed-sphere face (1–6) and the element's local 2D indices
`(ie, je)` within that face for the global element index `elem_idx`.
Reads `topology.elemorder`, which encodes the actual element layout —
either face-major `CartesianIndices((ne, ne, 6))` or a space-filling-curve
permutation `Vector{CartesianIndex{3}}`. This makes the mapping correct
for both regular and Gilbert-ordered topologies.
"""
function element_face_local_indices(topology::Topologies.Topology2D, elem_idx::Int)
    ci = topology.elemorder[elem_idx]
    return ci[3], ci[1], ci[2]   # (face, ie, je)
end

"""
    inverse_element_map(space, elem_idx, x) -> (ξ, η)

Given a 3D point `x` on the sphere known to lie inside element `elem_idx`,
return its element-local reference coordinates `(ξ, η) ∈ [-1, 1]²`.

Delegates to `ClimaCore.Meshes.reference_coordinates`, which handles both
`IntrinsicMap` (closed-form equiangular inversion) and `NormalizedBilinearMap`
(bilinear-invert against the four corner positions, the default for cubed
spheres). Both maps share the same vertex positions, so corner GLL nodes
agree under either; interior nodes differ, which is why a hand-rolled
equiangular-only inverse fails for the default mesh.
"""
function inverse_element_map(space::ClimaCore.Spaces.AbstractSpectralElementSpace,
                             elem_idx::Int, x)
    topology = Spaces.topology(space)
    mesh = Topologies.mesh(topology)
    face, ie, je = element_face_local_indices(topology, elem_idx)
    elem = CartesianIndex(ie, je, face)

    # ClimaCore's reference_coordinates expects a Cartesian123Point at the
    # mesh's radius scale; the inverse uses ζx/ζ0 ratios + bilinear_invert,
    # both of which are scale-equivariant, so a unit-sphere `x` works too.
    coord = ClimaCore.Geometry.Cartesian123Point(x[1], x[2], x[3])
    ξ1, ξ2 = ClimaCore.Meshes.reference_coordinates(mesh, elem, coord)
    return ξ1, ξ2
end

# ────────────────────────────────────────────────────────────────────────────
# Element Jacobian interpolator (Task 5, PDF Eq. 47)
# ────────────────────────────────────────────────────────────────────────────

"""
    element_jacobian_at(space, elem_idx, ξ, η) -> Float64

Evaluate the SE element's Jacobian at reference coordinates `(ξ, η)` by
Lagrange interpolation from the nodal weighted-Jacobian values stored on
`space` (PDF Eq. 47):

    Jᵉ(ξ, η) ≈ Σ_{p,q} Jᵉₚᵩ ϕₚ(ξ) ϕᵩ(η)

where `Jᵉₚᵩ = WJ[p, q, 1, elem_idx] / (wₚ wᵩ)` is the unweighted Jacobian
recovered from ClimaCore's `Spaces.weighted_jacobian` storage.
"""
function element_jacobian_at(space::ClimaCore.Spaces.AbstractSpectralElementSpace,
                             elem_idx::Int, ξ, η)
    qs = Spaces.quadrature_style(space)
    ξs, ws = Quadratures.quadrature_points(Float64, qs)
    Nq = length(ξs)

    WJ = parent(Spaces.weighted_jacobian(space))
    ϕξ = ConservativeRegridding.Lagrange.evaluate_all(ξs, ξ)
    ϕη = ConservativeRegridding.Lagrange.evaluate_all(ξs, η)

    Jᵉ = 0.0
    @inbounds for q in 1:Nq, p in 1:Nq
        Jₚᵩ = WJ[p, q, 1, elem_idx] / (ws[p] * ws[q])
        Jᵉ += Jₚᵩ * ϕξ[p] * ϕη[q]
    end
    return Jᵉ
end

# ────────────────────────────────────────────────────────────────────────────
# Principled B-accumulator (Task 6, PDF Eq. 48)
# ────────────────────────────────────────────────────────────────────────────

"""
    accumulate_principled_b(manifold, space, elem_idx, intersection_polygon;
                            triangle_quad_degree) -> Matrix{Float64}

Compute the principled `B(k, (e, i, j))` weights for a single source SE
element `e = elem_idx` and a single destination polygon `intersection_polygon`
(physical-space polygon, on the manifold). Returns an `Nq × Nq` matrix
`B[i, j]` such that

    B[i, j] ≈ ∫_{intersection_polygon} ϕᵢ(ξ) ϕⱼ(η) dA_phys       (PDF Eq. 48)

The Jacobian factor in PDF Eq. 18 cancels via change of variables: if we
integrate in physical space, `∫_{ref} ϕᵢϕⱼ Jᵉ dξ dη = ∫_{phys} ϕᵢ ϕⱼ dA`.
Approach: fan-triangulate the polygon from its centroid, apply a barycentric
Gauss rule on each triangle, evaluate the Lagrange basis at each quadrature
point (using `inverse_element_map` to obtain `(ξ, η)`).
"""
function accumulate_principled_b(
    manifold::GOCore.Manifold,
    space::ClimaCore.Spaces.AbstractSpectralElementSpace,
    elem_idx::Int,
    intersection_polygon;
    triangle_quad_degree::Int,
)
    qs = Spaces.quadrature_style(space)
    ξs, _ = Quadratures.quadrature_points(Float64, qs)
    Nq = length(ξs)

    bary, w = ConservativeRegridding.TriangleQuadrature.reference_rule(triangle_quad_degree)

    B = zeros(Float64, Nq, Nq)

    ring = GI.getexterior(intersection_polygon)
    npts = GI.npoint(ring) - 1   # exclude closing point
    npts < 3 && return B
    verts = [GI.getpoint(ring, k) for k in 1:npts]

    centroid = polygon_centroid(manifold, verts)

    for k in 1:npts
        v₁ = verts[k]
        v₂ = verts[mod1(k + 1, npts)]
        Aₜ = spherical_triangle_area(manifold, centroid, v₁, v₂)
        Aₜ ≤ 0 && continue

        for (λ, wᵧ) in zip(bary, w)
            xᵧ = bary_to_physical(manifold, λ, centroid, v₁, v₂)

            ξ, η = inverse_element_map(space, elem_idx, xᵧ)
            ϕξ = ConservativeRegridding.Lagrange.evaluate_all(ξs, ξ)
            ϕη = ConservativeRegridding.Lagrange.evaluate_all(ξs, η)

            # Reference rule's weights sum to 1/2 (ref triangle area); to map
            # onto a physical triangle of area Aₜ, scale by Aₜ/(1/2) = 2 Aₜ.
            wAᵧ = wᵧ * 2 * Aₜ

            @inbounds for j in 1:Nq, i in 1:Nq
                B[i, j] += wAᵧ * ϕξ[i] * ϕη[j]
            end
        end
    end
    return B
end

# Polygon centroid: arithmetic mean of vertices, projected to sphere on Spherical.
function polygon_centroid(::GOCore.Spherical, verts)
    n = length(verts)
    sx = sum(GI.x(v) for v in verts) / n
    sy = sum(GI.y(v) for v in verts) / n
    sz = sum(GI.z(v) for v in verts) / n
    norm = sqrt(sx^2 + sy^2 + sz^2)
    return (sx / norm, sy / norm, sz / norm)
end

function polygon_centroid(::GOCore.Planar, verts)
    n = length(verts)
    sx = sum(GI.x(v) for v in verts) / n
    sy = sum(GI.y(v) for v in verts) / n
    return (sx, sy)
end

# Map barycentric (λ₁, λ₂, λ₃) on triangle (c, v₁, v₂) to a physical point.
# On the sphere, project the linear combination back to the unit sphere.
function bary_to_physical(::GOCore.Spherical, λ, c, v₁, v₂)
    cx, cy, cz = c[1], c[2], c[3]
    x₁, y₁, z₁ = GI.x(v₁), GI.y(v₁), GI.z(v₁)
    x₂, y₂, z₂ = GI.x(v₂), GI.y(v₂), GI.z(v₂)
    px = λ[1] * cx + λ[2] * x₁ + λ[3] * x₂
    py = λ[1] * cy + λ[2] * y₁ + λ[3] * y₂
    pz = λ[1] * cz + λ[2] * z₁ + λ[3] * z₂
    norm = sqrt(px^2 + py^2 + pz^2)
    return (px / norm, py / norm, pz / norm)
end

function bary_to_physical(::GOCore.Planar, λ, c, v₁, v₂)
    cx, cy = c[1], c[2]
    x₁, y₁ = GI.x(v₁), GI.y(v₁)
    x₂, y₂ = GI.x(v₂), GI.y(v₂)
    return (λ[1] * cx + λ[2] * x₁ + λ[3] * x₂,
            λ[1] * cy + λ[2] * y₁ + λ[3] * y₂)
end

# Spherical triangle area via GeometryOps (Girard's theorem internally). Build
# the ring out of `UnitSphericalPoint`s so GeometryOps treats them as 3D
# unit-sphere coordinates rather than (lon, lat, _) geographic tuples.
function spherical_triangle_area(::GOCore.Spherical, p₁, p₂, p₃)
    q₁ = UnitSphericalPoint(p₁[1], p₁[2], p₁[3])
    q₂ = UnitSphericalPoint(p₂[1], p₂[2], p₂[3])
    q₃ = UnitSphericalPoint(p₃[1], p₃[2], p₃[3])
    poly = GI.Polygon([GI.LinearRing([q₁, q₂, q₃, q₁])])
    return GO.area(GO.Spherical(), poly)
end

function spherical_triangle_area(::GOCore.Planar, p₁, p₂, p₃)
    return ConservativeRegridding.TriangleQuadrature.planar_triangle_area(
        ((p₁[1], p₁[2]), (p₂[1], p₂[2]), (p₃[1], p₃[2]))
    )
end

"""
    se_field_to_vec(field)

Convert a ClimaCore field to a flat vector of nodal values.
Same ordering as [`se_node_positions`](@ref).
"""
function se_field_to_vec(field)
    return _flat_nodal_data(parent(Fields.field_values(field)))
end

"""
    vec_to_se_field!(field, v::AbstractVector) → field

Copy values from a flat nodal vector back into a ClimaCore field.
Inverse of [`se_field_to_vec`](@ref).
"""
function vec_to_se_field!(field, v::AbstractVector)
    p = parent(Fields.field_values(field))
    Nq = size(p, 1)
    if ndims(p) == 4
        Nh = size(p, 4)
        view(p, :, :, 1, :) .= reshape(v, Nq, Nq, Nh)
    elseif ndims(p) == 5
        Nh = size(p, 5)
        view(p, :, :, 1, 1, :) .= reshape(v, Nq, Nq, Nh)
    end
    return field
end

## Shared helpers for SE regridder construction

"""
    _get_candidate_pairs(manifold, se_tree, cell_tree, threaded)

Get all candidate (SE element, cell) pairs for regridding.
Uses a dual depth-first search to find all pairs of cells/elements that may intersect.
"""
function _get_candidate_pairs(manifold, se_tree, cell_tree, threaded)
    _threaded = GOCore.booltype(threaded)
    predicate_f = manifold isa GOCore.Spherical ?
        GO.UnitSpherical._intersects : Extents.intersects
    return ConservativeRegridding.get_all_candidate_pairs(
        _threaded, predicate_f, se_tree, cell_tree
    )
end

## Regridder constructors

# SE source → FV destination (principled polygon-intersection, PDF Appendix A)
function ConservativeRegridding.Regridder(
    manifold::M, dst, src::ClimaCore.Spaces.AbstractSpectralElementSpace;
    triangle_quad_degree::Union{Int, Nothing} = nothing,
    threaded = true,
    kwargs...
) where {M <: GOCore.Manifold}
    return se_to_fv_principled(manifold, dst, src; threaded, triangle_quad_degree, kwargs...)
end

function se_to_fv_principled(manifold, dst, src; threaded, triangle_quad_degree, kwargs...)
    se_tree = Trees.treeify(manifold, src)
    fv_tree = Trees.treeify(manifold, dst)

    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(src))
    Nh = Meshes.nelements(Topologies.mesh(Spaces.topology(src)))
    N_nodes = Nq^2 * Nh
    N_fv = prod(Trees.ncells(fv_tree))

    triangle_quad_degree = something(triangle_quad_degree, 2 * (Nq - 1))

    candidate_pairs = _get_candidate_pairs(manifold, se_tree, fv_tree, threaded)

    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for (elem_idx, cell_idx) in candidate_pairs
        elem_poly = Trees.getcell(se_tree, elem_idx)
        cell_poly = Trees.getcell(fv_tree, cell_idx)

        intersection_result = if manifold isa GOCore.Spherical
            GO.intersection(GO.ConvexConvexSutherlandHodgman(manifold),
                            elem_poly, cell_poly; target = GI.PolygonTrait())
        else
            GO.intersection(GO.FosterHormannClipping(GO.Planar()),
                            elem_poly, cell_poly; target = GI.PolygonTrait())
        end

        # GO.intersection may return a single Polygon, a Vector{Polygon}, or
        # nothing for an empty intersection. Normalize to an iterable.
        intersection_polys = if intersection_result === nothing
            ()
        elseif intersection_result isa AbstractVector
            intersection_result
        else
            (intersection_result,)   # single Polygon
        end

        for ipoly in intersection_polys
            B = accumulate_principled_b(manifold, src, elem_idx, ipoly;
                                        triangle_quad_degree)
            offset = (elem_idx - 1) * Nq^2
            for j in 1:Nq, i in 1:Nq
                Bᵢⱼ = B[i, j]
                Bᵢⱼ == 0 && continue
                push!(rows, cell_idx)
                push!(cols, offset + (j - 1) * Nq + i)
                push!(vals, Bᵢⱼ)
            end
        end
    end

    weight_matrix = SparseArrays.sparse(rows, cols, vals, N_fv, N_nodes)
    dst_areas = ConservativeRegridding.areas(manifold, dst, fv_tree)

    return ConservativeRegridding.SEtoFVRegridder(
        weight_matrix, dst_areas, zeros(N_fv), zeros(N_nodes),
    )
end

# FV source → SE destination (per-element L2 projection)
function ConservativeRegridding.Regridder(
    manifold::M, dst::ClimaCore.Spaces.AbstractSpectralElementSpace, src;
    triangle_quad_degree::Union{Int, Nothing} = nothing,
    threaded = true,
    kwargs...
) where {M <: GOCore.Manifold}
    return fv_to_se_l2_projection(manifold, dst, src; threaded, triangle_quad_degree, kwargs...)
end

# Disambiguate SE → SE: no longer supported on this branch.
function ConservativeRegridding.Regridder(
    manifold::M,
    dst::ClimaCore.Spaces.AbstractSpectralElementSpace,
    src::ClimaCore.Spaces.AbstractSpectralElementSpace;
    kwargs...
) where {M <: GOCore.Manifold}
    error("SE → SE regridding is not supported. Use SE → FV → SE via two regridders.")
end

"""
    compute_local_mass_matrix(space, elem_idx) -> Matrix{Float64}

Build the dense `Nq² × Nq²` mass matrix for SE element `elem_idx`,

    M^{e}_{(a,b),(i,j)} = ∫_{e} ϕₐ(ξ) ϕᵦ(η) ϕᵢ(ξ) ϕⱼ(η) Jᵉ(ξ,η) dξ dη ,

with the row/col flattening `idx = (b - 1) * Nq + a`. The integrand is a
polynomial of bidegree `(2(Nq-1), 2(Nq-1))` in (ξ, η) (the basis-function
product) plus a polynomial of bidegree `(Nq-1, Nq-1)` from the Lagrange
interpolation of `Jᵉ` from the SE nodal values — total bidegree
`(3(Nq-1), 3(Nq-1))`. A tensor-product GLL rule with `Nq + 2` points per
direction has 1D exactness `2(Nq+2) - 3 = 2Nq + 1`, which suffices.
"""
function compute_local_mass_matrix(
    space::ClimaCore.Spaces.AbstractSpectralElementSpace, elem_idx::Int,
)
    qs = Spaces.quadrature_style(space)
    ξs_se, _ = Quadratures.quadrature_points(Float64, qs)
    Nq = length(ξs_se)

    # GLL with Nq+2 points exactly integrates the polynomial part.
    qs_q = Quadratures.GLL{Nq + 2}()
    ξs_q, ws_q = Quadratures.quadrature_points(Float64, qs_q)
    Nq_q = length(ξs_q)

    Nq² = Nq * Nq
    M = zeros(Nq², Nq²)

    @inbounds for q in 1:Nq_q, p in 1:Nq_q
        ξ = ξs_q[p]
        η = ξs_q[q]
        wξ = ws_q[p]
        wη = ws_q[q]

        ϕξ = ConservativeRegridding.Lagrange.evaluate_all(ξs_se, ξ)
        ϕη = ConservativeRegridding.Lagrange.evaluate_all(ξs_se, η)
        Jᵉ = element_jacobian_at(space, elem_idx, ξ, η)

        wJ = wξ * wη * Jᵉ
        for jb in 1:Nq, ja in 1:Nq
            row = (jb - 1) * Nq + ja
            ϕaξ = ϕξ[ja]
            ϕbη = ϕη[jb]
            for jj in 1:Nq, ii in 1:Nq
                col = (jj - 1) * Nq + ii
                M[row, col] += wJ * ϕaξ * ϕξ[ii] * ϕbη * ϕη[jj]
            end
        end
    end
    return M
end

"""
    fv_to_se_l2_projection(manifold, dst, src; ...)

Per-element L2 projection FV → SE. Same `B` accumulation as the principled
path, but the per-element rows are then multiplied by the *full* local mass
matrix inverse `(M^{e})^{-1}` (rather than divided by the lumped diagonal
`Wᵉᵢⱼ` as in PDF Eq. 30). This preserves constants exactly and is higher-
order accurate: for any `f_src` that is exactly representable as a constant
in each FV cell, the projection `f_dst = (M^{e})^{-1} (Bᵀ f_src)|_e` is the
optimal L2 fit on the SE basis over element `e`.
"""
function fv_to_se_l2_projection(manifold, dst, src;
                                threaded, triangle_quad_degree, kwargs...)
    se_tree = Trees.treeify(manifold, dst)
    fv_tree = Trees.treeify(manifold, src)

    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(dst))
    Nh = Meshes.nelements(Topologies.mesh(Spaces.topology(dst)))
    N_nodes = Nq^2 * Nh
    N_fv = prod(Trees.ncells(fv_tree))

    triangle_quad_degree = something(triangle_quad_degree, 2 * (Nq - 1))

    candidate_pairs = _get_candidate_pairs(manifold, se_tree, fv_tree, threaded)

    # Accumulate B per element, keyed by destination cell, value = Nq² vector.
    elem_b_dicts = [Dict{Int, Vector{Float64}}() for _ in 1:Nh]

    for (elem_idx, cell_idx) in candidate_pairs
        elem_poly = Trees.getcell(se_tree, elem_idx)
        cell_poly = Trees.getcell(fv_tree, cell_idx)

        intersection_result = if manifold isa GOCore.Spherical
            GO.intersection(GO.ConvexConvexSutherlandHodgman(manifold),
                            elem_poly, cell_poly; target = GI.PolygonTrait())
        else
            GO.intersection(GO.FosterHormannClipping(GO.Planar()),
                            elem_poly, cell_poly; target = GI.PolygonTrait())
        end

        intersection_polys = if intersection_result === nothing
            ()
        elseif intersection_result isa AbstractVector
            intersection_result
        else
            (intersection_result,)
        end

        for ipoly in intersection_polys
            B = accumulate_principled_b(manifold, dst, elem_idx, ipoly;
                                        triangle_quad_degree)
            d = elem_b_dicts[elem_idx]
            v = get!(d, cell_idx) do
                zeros(Nq^2)
            end
            for jb in 1:Nq, ja in 1:Nq
                Bᵢⱼ = B[ja, jb]
                Bᵢⱼ == 0 && continue
                v[(jb - 1) * Nq + ja] += Bᵢⱼ
            end
        end
    end

    # For each element, solve Mᵉ f_dst = b for each cell column, emit COO.
    #
    # Mᵉ comes from tensor-product GLL on the reference square but B from fan
    # triangulation on the great-circle physical polygon, so the two
    # quadratures integrate over slightly different domains. We rescale each
    # row of Mᵉ so that its row sum equals the column-sum of B for that
    # element — what Mᵉ·1 must equal by partition of unity if both sides used
    # identical quadrature. This forces exact constant preservation.
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for elem_idx in 1:Nh
        d = elem_b_dicts[elem_idx]
        isempty(d) && continue

        Mᵉ = compute_local_mass_matrix(dst, elem_idx)

        b_full = zeros(Nq^2)
        for b_vec in values(d)
            b_full .+= b_vec
        end

        row_sums = vec(sum(Mᵉ; dims=2))
        for r in 1:(Nq^2)
            row_sums[r] == 0 && continue
            scale = b_full[r] / row_sums[r]
            @inbounds for c in 1:(Nq^2)
                Mᵉ[r, c] *= scale
            end
        end

        Mᵉ_inv = inv(Mᵉ)

        offset = (elem_idx - 1) * Nq^2
        for (cell_idx, b_vec) in d
            new_col = Mᵉ_inv * b_vec
            for n in 1:Nq^2
                v = new_col[n]
                v == 0 && continue
                push!(rows, offset + n)
                push!(cols, cell_idx)
                push!(vals, v)
            end
        end
    end

    weight_matrix = SparseArrays.sparse(rows, cols, vals, N_nodes, N_fv)

    return ConservativeRegridding.FVtoSERegridder(
        weight_matrix, zeros(N_nodes), zeros(N_fv),
    )
end

## Field-level regrid! convenience interface

"""
    regrid!(dst::AbstractVector, regridder::SEtoFVRegridder, src::ClimaCore.Fields.Field)

Remap a ClimaCore spectral element `Field` to a flat FV vector, converting the field
to a flat nodal vector internally.
"""
function ConservativeRegridding.regrid!(
    dst::AbstractVector,
    regridder::ConservativeRegridding.SEtoFVRegridder,
    src::ClimaCore.Fields.Field,
)
    return ConservativeRegridding.regrid!(dst, regridder, se_field_to_vec(src))
end

"""
    regrid!(dst::ClimaCore.Fields.Field, regridder::FVtoSERegridder, src::AbstractVector)

Remap a flat FV vector into a ClimaCore spectral element `Field`, writing the
result back into `dst` in-place. The per-element L2 projection is not
automatically continuous, so we apply weighted DSS (mass-conserving) to
reconcile shared boundary nodes.
"""
function ConservativeRegridding.regrid!(
    dst::ClimaCore.Fields.Field,
    regridder::ConservativeRegridding.FVtoSERegridder,
    src::AbstractVector,
)
    dst_vec = regridder.dst_temp
    ConservativeRegridding.regrid!(dst_vec, regridder, src)
    vec_to_se_field!(dst, dst_vec)
    Spaces.weighted_dss!(dst)
    return dst
end

end
