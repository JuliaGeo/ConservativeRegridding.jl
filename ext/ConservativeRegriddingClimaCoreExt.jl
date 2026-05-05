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

"""
    _unique_node_weights(space) → Vector{Float64}

Return the assembled Jacobian integration weight for each unique SE node,
in the same iteration order as `Spaces.unique_nodes(space)`.

For a node shared by `k` elements, the per-element `WJ` represents only `1/k` of
the physical node's total contribution.  Dividing by the precomputed DSS weight
(`J / sum(collocated J)`, stored in `space.grid.dss_weights`) recovers the full
assembled value across all elements for a shared node:

    assembled_WJ[e,i,j] = WJ[e,i,j] / dss_weight[e,i,j]

Summing this assembled WJ over all unique nodes gives the total quadrature weight
(≈ sphere area), ensuring global conservation.
"""
function _unique_node_weights(space)
    wj_data  = parent(Spaces.weighted_jacobian(space))
    dss_data = parent(space.grid.dss_weights)
    return Float64[
        wj_data[i, j, 1, elem_idx] / dss_data[i, j, 1, elem_idx]
        for ((i, j), elem_idx) in Spaces.unique_nodes(space)
    ]
end

# ────────────────────────────────────────────────────────────────────────────
# Inverse element map for the equiangular cubed sphere (Task 4)
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

"""
    _point_in_convex_spherical_polygon(point, ring)

Check whether a `UnitSphericalPoint` lies strictly inside a convex polygon on the
unit sphere, using the cross-product sign-consistency test. For a convex polygon on the
unit sphere, P is inside if and only if it lies on the same side of every edge of the polygon.

For an edge from vertex A to vertex B, the signed distance of P from that edge's
great-circle plane is the scalar triple product: `A x B · P`,
which is positive if P is on the "left" of the directed edge A→B, negative on the "right".
For a consistently-wound convex polygon, all edges should give the same sign for an interior point.
"""
function _point_in_convex_spherical_polygon(point, ring)
    n = GI.npoint(ring) - 1  # unique vertices (exclude closing point)
    n < 3 && return false

    first_sign = 0
    for k in 1:n
        A = GI.getpoint(ring, k)
        B = GI.getpoint(ring, k + 1)
        tp = _triple_product(A, B, point)
        s = sign(tp)
        if s != 0
            if first_sign == 0
                first_sign = s
            elseif s != first_sign
                return false
            end
        end
    end
    return first_sign != 0
end

function get_element_centroids(space)
    # Get the indices of the vertices of the elements, in clockwise order for each element
    Nh = Meshes.nelements(space.grid.topology.mesh)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    vertex_inds = [
        CartesianIndex(i, j, 1, 1, e) # f and v are 1 for SpectralElementSpace2D
        for e in 1:Nh, (i, j) in [(1, 1), (Nq, Nq)]
    ] # repeat the first coordinate pair at the end

    # Get the lat and lon at each vertex index
    coords = Fields.coordinate_field(space)
    lonlat_to_usp = GO.UnitSpherical.UnitSphereFromGeographic()
    centroids = map(eachslice(vertex_inds; dims = 1)) do (ind1, ind2)
        coord1 = (Fields.field_values(coords.long)[ind1], Fields.field_values(coords.lat)[ind1])
        coord2 = (Fields.field_values(coords.long)[ind2], Fields.field_values(coords.lat)[ind2])
        usp_coord1, usp_coord2 = lonlat_to_usp.((coord1, coord2))
        return GO.UnitSpherical.slerp(usp_coord1, usp_coord2, 0.5)
    end
    return centroids
end

### These functions are used to facilitate storing a single value per element on a field
### rather than one value per node.
"""
    _triple_product(a, b, c)

Compute the triple product of three vectors.

Accumulate the sign of the first non-zero edge test, then return false the
moment any subsequent edge gives the opposite sign. If all edges give the
same sign (or zero, for a point exactly on an edge), the point is inside.
"""
@inline function _triple_product(a, b, c)
    ax, ay, az = a[1], a[2], a[3]
    bx, by, bz = b[1], b[2], b[3]
    cx, cy, cz = c[1], c[2], c[3]
    return cx * (ay*bz - az*by) + cy * (az*bx - ax*bz) + cz * (ax*by - ay*bx)
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

"""
    _find_nodes_in_cells(candidate_pairs, cell_tree, node_positions, Nq)

For each candidate (SE element, cell) pair, check which SE nodes lie inside the
cell polygon. Iterates over ALL `Nq²` nodes of each element.  Each flat node index
is matched to at most one cell (the first match wins).

Returns a `Vector{Tuple{Int,Int}}` of `(node_idx, cell_idx)` pairs.

Used for FV→SE regridding, where every node occurrence (including shared boundary
nodes) should independently receive the containing FV cell's value.
"""
function _find_nodes_in_cells(candidate_pairs, cell_tree, node_positions, Nq::Int)
    pairs = Tuple{Int,Int}[]
    matched = Set{Int}()

    for (se_elem_idx, cell_idx) in candidate_pairs
        cell_polygon = Trees.getcell(cell_tree, cell_idx)
        ring = GI.getexterior(cell_polygon)

        node_offset = (se_elem_idx - 1) * Nq^2
        for jj in 1:Nq, ii in 1:Nq
            node_idx = node_offset + (jj - 1) * Nq + ii
            node_idx in matched && continue
            if _point_in_convex_spherical_polygon(node_positions[node_idx], ring)
                push!(pairs, (node_idx, cell_idx))
                push!(matched, node_idx)
            end
        end
    end

    return pairs
end

"""
    _add_intersection_fallback!(weight_matrix, candidate_pairs, zero_rows, manifold, se_tree, cell_tree, node_weights, Nq, N_cells, N_nodes)

For destination cells with zero node contribution (typically when the destination cells are
much finer than the SE nodal spacing), fall back to intersection-area-weighted element
averages for those rows. Adds entries to the weight matrix for those rows.

Each node `(e,i,j)` in a SE element `e` that intersects destination cell `k` receives weight:
    `int_area(e, k) * W[e,i,j] / ∑_{i',j'} W[e,i',j']`

!!! note "Trade-off (PDF §2 Case 2, Option 2)"
    The primary scheme places each unique node's *assembled* WJ at one
    representative `flat_idx`; this fallback adds extra entries for *every*
    `(e,i,j)` of every element overlapping a zero row, using *raw* per-element
    WJ. For shared boundary nodes this means the source mass is double-counted
    (assembled in the primary path **and** raw-per-element in the fallback path),
    so the simplified scheme's global conservation is broken by an amount
    proportional to the area of zero-row destination cells. This is the
    documented price paid to avoid leaving FV cells with no value at all.
    The principled path (`method=:polygon_intersection`, default) does not use
    this fallback and is exactly conservative.

This distributes the intersection area proportionally to the Jacobian weights within the element.
"""
function _add_intersection_fallback!(
    weight_matrix, candidate_pairs, zero_rows,
    manifold, se_tree, cell_tree, node_weights, Nq, N_cells, N_nodes
)
    isempty(zero_rows) && return weight_matrix

    zero_row_set = Set(zero_rows)
    intersection_op = ConservativeRegridding.DefaultIntersectionOperator(manifold)

    # Filter to only pairs involving zero-contribution destination cells, then
    # reuse the existing intersection area computation from intersection_areas.jl.
    # Candidate pairs are (se_elem_idx, cell_idx); compute_intersection_areas
    # treats the first index as src (se_tree) and second as dst (cell_tree).
    fallback_pairs = filter(((_, cell_idx),) -> cell_idx in zero_row_set, candidate_pairs)
    isempty(fallback_pairs) && return weight_matrix

    se_idxs, cell_idxs, int_areas = ConservativeRegridding.compute_intersection_areas(
        manifold, intersection_op, cell_tree, se_tree, fallback_pairs
    )

    fb_rows = Int[]
    fb_cols = Int[]
    fb_vals = Float64[]

    for (se_elem_idx, cell_idx, int_area) in zip(se_idxs, cell_idxs, int_areas)
        elem_offset = (se_elem_idx - 1) * Nq^2
        elem_area   = sum(node_weights[elem_offset+1 : elem_offset+Nq^2])
        for jj in 1:Nq, ii in 1:Nq
            node_idx = elem_offset + (jj - 1) * Nq + ii
            push!(fb_rows, cell_idx)
            push!(fb_cols, node_idx)
            push!(fb_vals, int_area * node_weights[node_idx] / elem_area)
        end
    end

    isempty(fb_rows) && return weight_matrix
    fb_matrix = SparseArrays.sparse(fb_rows, fb_cols, fb_vals, N_cells, N_nodes)
    return weight_matrix + fb_matrix
end

## Regridder constructors

# SE source → FV destination
function ConservativeRegridding.Regridder(
    manifold::M, dst, src::ClimaCore.Spaces.AbstractSpectralElementSpace;
    method::Symbol = :polygon_intersection,
    triangle_quad_degree::Union{Int, Nothing} = nothing,
    threaded = true,
    kwargs...
) where {M <: GOCore.Manifold}
    if method == :node_in_polygon
        return se_to_fv_simplified(manifold, dst, src; threaded, kwargs...)
    elseif method == :polygon_intersection
        return se_to_fv_principled(manifold, dst, src; threaded, triangle_quad_degree, kwargs...)
    else
        error("Unknown method `$method`. Use :polygon_intersection (default) or :node_in_polygon.")
    end
end

function se_to_fv_simplified(manifold, dst, src; threaded, kwargs...)
    se_tree = Trees.treeify(manifold, src)
    fv_tree = Trees.treeify(manifold, dst)

    node_positions = se_node_positions(src)
    unique_wj = _unique_node_weights(src)
    per_elem_weights = se_node_weights(src)

    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(src))
    Nh = Meshes.nelements(Topologies.mesh(Spaces.topology(src)))
    N_nodes = Nq^2 * Nh
    N_fv = prod(Trees.ncells(fv_tree))

    candidate_pairs = _get_candidate_pairs(manifold, se_tree, fv_tree, threaded)
    elem_to_cells = Dict{Int, Vector{Int}}()
    for (se_elem_idx, cell_idx) in candidate_pairs
        push!(get!(elem_to_cells, se_elem_idx, Int[]), cell_idx)
    end

    rows, cols, vals = Int[], Int[], Float64[]
    for (k, ((i, j), elem_idx)) in enumerate(Spaces.unique_nodes(src))
        flat_idx = (elem_idx - 1) * Nq^2 + (j - 1) * Nq + i
        pos = node_positions[flat_idx]
        wj  = unique_wj[k]
        for cell_idx in get(elem_to_cells, elem_idx, Int[])
            if _point_in_convex_spherical_polygon(pos, GI.getexterior(Trees.getcell(fv_tree, cell_idx)))
                push!(rows, cell_idx)
                push!(cols, flat_idx)
                push!(vals, wj)
                break
            end
        end
    end
    weight_matrix = SparseArrays.sparse(rows, cols, vals, N_fv, N_nodes)

    zero_rows = findall(iszero, vec(sum(weight_matrix; dims=2)))
    weight_matrix = _add_intersection_fallback!(
        weight_matrix, candidate_pairs, zero_rows,
        manifold, se_tree, fv_tree, per_elem_weights, Nq, N_fv, N_nodes
    )

    dst_areas = ConservativeRegridding.areas(manifold, dst, fv_tree)

    return ConservativeRegridding.SEtoFVRegridder(
        weight_matrix, dst_areas, zeros(N_fv), zeros(N_nodes),
    )
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

# FV source → SE destination
#
# `:polygon_intersection` (default): per-element L2 projection. The principled
# `B(k,(e,i,j)) = ∫_{k∩e} ϕᵢϕⱼ dA` is built per (cell, element) intersection,
# then the per-element rows of the weight matrix are multiplied by the
# *full* local mass matrix inverse `(M^{e})^{-1}` (not divided by the lumped
# diagonal Wᵉᵢⱼ as PDF Eq. 30 does). This preserves constants exactly via
# partition of unity (`M^{e} 1 = Bᵀ 1` for any constant source) and is
# higher-order accurate.
#
# `:node_in_polygon`: simplified, every SE node receives the value of its
# containing FV cell. Constants preserved exactly, output is automatically
# DSS-consistent.
function ConservativeRegridding.Regridder(
    manifold::M, dst::ClimaCore.Spaces.AbstractSpectralElementSpace, src;
    method::Symbol = :polygon_intersection,
    triangle_quad_degree::Union{Int, Nothing} = nothing,
    threaded = true,
    kwargs...
) where {M <: GOCore.Manifold}
    if method == :node_in_polygon
        return fv_to_se_simplified(manifold, dst, src; threaded, kwargs...)
    elseif method == :polygon_intersection
        return fv_to_se_l2_projection(manifold, dst, src; threaded, triangle_quad_degree, kwargs...)
    else
        error("Unknown method `$method`. Use :polygon_intersection (default) or :node_in_polygon.")
    end
end

function fv_to_se_simplified(manifold, dst, src; threaded, kwargs...)
    se_tree = Trees.treeify(manifold, dst)
    fv_tree = Trees.treeify(manifold, src)

    node_positions = se_node_positions(dst)

    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(dst))
    Nh = Meshes.nelements(Topologies.mesh(Spaces.topology(dst)))
    N_nodes = Nq^2 * Nh
    N_fv = prod(Trees.ncells(fv_tree))

    candidate_pairs = _get_candidate_pairs(manifold, se_tree, fv_tree, threaded)
    node_cell_pairs = _find_nodes_in_cells(candidate_pairs, fv_tree, node_positions, Nq)
    node_idxs = first.(node_cell_pairs)
    cell_idxs  = last.(node_cell_pairs)

    weight_matrix = SparseArrays.sparse(
        node_idxs, cell_idxs, ones(length(node_idxs)), N_nodes, N_fv
    )

    return ConservativeRegridding.FVtoSERegridder(
        weight_matrix, zeros(N_nodes), zeros(N_fv), nothing,
    )
end

function fv_to_se_principled(manifold, dst, src; threaded, triangle_quad_degree, kwargs...)
    se_tree = Trees.treeify(manifold, dst)
    fv_tree = Trees.treeify(manifold, src)

    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(dst))
    Nh = Meshes.nelements(Topologies.mesh(Spaces.topology(dst)))
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
            offset = (elem_idx - 1) * Nq^2
            for j in 1:Nq, i in 1:Nq
                Bᵢⱼ = B[i, j]
                Bᵢⱼ == 0 && continue
                push!(rows, offset + (j - 1) * Nq + i)
                push!(cols, cell_idx)
                push!(vals, Bᵢⱼ)
            end
        end
    end

    weight_matrix = SparseArrays.sparse(rows, cols, vals, N_nodes, N_fv)

    # Per-element raw WJᵉᵢⱼ for normalization (PDF Eq. 30 divides by Wᵉᵢⱼ).
    per_node_wj = se_node_weights(dst)   # length N_nodes, in flat (e,i,j) order
    inv_node_weights = [w == 0 ? 0.0 : 1 / w for w in per_node_wj]

    return ConservativeRegridding.FVtoSERegridder(
        weight_matrix, zeros(N_nodes), zeros(N_fv), inv_node_weights,
    )
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

    # For each element, solve M^{e} f_dst = b for each cell column, emit COO.
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for elem_idx in 1:Nh
        d = elem_b_dicts[elem_idx]
        isempty(d) && continue

        M = compute_local_mass_matrix(dst, elem_idx)
        M_inv = inv(M)

        offset = (elem_idx - 1) * Nq^2
        for (cell_idx, b_vec) in d
            new_col = M_inv * b_vec
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
        weight_matrix, zeros(N_nodes), zeros(N_fv), nothing,   # no per-node 1/W needed
    )
end

# SE source → SE destination
function ConservativeRegridding.Regridder(
    manifold::M,
    dst::ClimaCore.Spaces.AbstractSpectralElementSpace,
    src::ClimaCore.Spaces.AbstractSpectralElementSpace;
    threaded = true,
    kwargs...
) where {M <: GOCore.Manifold}
    # Convert the SE source and destination grids into trees
    src_se_tree = Trees.treeify(manifold, src)
    dst_se_tree = Trees.treeify(manifold, dst)

    # Get the SE node positions and weights as flat vectors
    src_node_positions = se_node_positions(src)
    unique_src_wj = _unique_node_weights(src)
    per_elem_src_weights = se_node_weights(src)
    dst_node_weights = se_node_weights(dst)

    # Get the number of SE source and destination nodes
    Nq_src = Quadratures.degrees_of_freedom(Spaces.quadrature_style(src))
    Nh_src = Meshes.nelements(Topologies.mesh(Spaces.topology(src)))
    N_src_nodes = Nq_src^2 * Nh_src

    Nq_dst = Quadratures.degrees_of_freedom(Spaces.quadrature_style(dst))
    Nh_dst = Meshes.nelements(Topologies.mesh(Spaces.topology(dst)))
    N_dst_nodes = Nq_dst^2 * Nh_dst

    # Get all source SE nodes that lie in the destination SE elements we're remapping to
    candidate_pairs = _get_candidate_pairs(manifold, src_se_tree, dst_se_tree, threaded)
    src_elem_to_dst_elems = Dict{Int, Vector{Int}}()
    for (src_elem_idx, dst_elem_idx) in candidate_pairs
        push!(get!(src_elem_to_dst_elems, src_elem_idx, Int[]), dst_elem_idx)
    end

    # The weights are the source SE node weights for the source SE nodes in the destination
    # SE elements we're remapping to
    rows, cols, vals = Int[], Int[], Float64[]
    for (k, ((i, j), elem_idx)) in enumerate(Spaces.unique_nodes(src))
        flat_idx = (elem_idx - 1) * Nq_src^2 + (j - 1) * Nq_src + i
        pos = src_node_positions[flat_idx]
        wj  = unique_src_wj[k]
        for dst_elem_idx in get(src_elem_to_dst_elems, elem_idx, Int[])
            if _point_in_convex_spherical_polygon(pos, GI.getexterior(Trees.getcell(dst_se_tree, dst_elem_idx)))
                push!(rows, dst_elem_idx); push!(cols, flat_idx); push!(vals, wj)
                break
            end
        end
    end
    weight_matrix = SparseArrays.sparse(rows, cols, vals, Nh_dst, N_src_nodes)

    # Find all rows of the weight matrix that have zero sum (i.e. no source SE nodes in the dest SE element)
    # and add the fallback intersection-area-weighted element averages to those rows
    zero_rows = findall(iszero, vec(sum(weight_matrix; dims=2)))
    weight_matrix = _add_intersection_fallback!(
        weight_matrix, candidate_pairs, zero_rows,
        manifold, src_se_tree, dst_se_tree, per_elem_src_weights, Nq_src, Nh_dst, N_src_nodes
    )

    # Get the areas of the destination SE elements, which are used for normalization
    Nq_dst2 = Nq_dst^2
    dst_element_areas = [
        sum(dst_node_weights[(e-1)*Nq_dst2+1 : e*Nq_dst2]) for e in 1:Nh_dst
    ]

    # Allocate temporary arrays for the regridding operation
    dst_temp = zeros(N_dst_nodes)
    src_temp = zeros(N_src_nodes)

    return ConservativeRegridding.SEtoSERegridder(
        weight_matrix, dst_element_areas, Nq_dst, dst_temp, src_temp
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
result back into `dst` in-place.

For the principled path (`regridder.inv_node_weights !== nothing`) the per-element
output is not automatically continuous, so we apply weighted DSS to reconcile
shared boundary nodes (mass-conserving). The simplified path is already
DSS-consistent because all duplicates of a physical node map to the same FV cell.
"""
function ConservativeRegridding.regrid!(
    dst::ClimaCore.Fields.Field,
    regridder::ConservativeRegridding.FVtoSERegridder,
    src::AbstractVector,
)
    dst_vec = regridder.dst_temp
    ConservativeRegridding.regrid!(dst_vec, regridder, src)
    vec_to_se_field!(dst, dst_vec)
    if regridder.inv_node_weights !== nothing
        Spaces.weighted_dss!(dst)
    end
    return dst
end

"""
    regrid!(dst::ClimaCore.Fields.Field, regridder::SEtoSERegridder, src::ClimaCore.Fields.Field)

Remap one ClimaCore spectral element `Field` to another, converting both fields to
flat nodal vectors internally and writing the result back into `dst` in-place.
"""
function ConservativeRegridding.regrid!(
    dst::ClimaCore.Fields.Field,
    regridder::ConservativeRegridding.SEtoSERegridder,
    src::ClimaCore.Fields.Field,
)
    dst_vec = regridder.dst_temp
    ConservativeRegridding.regrid!(dst_vec, regridder, se_field_to_vec(src))
    vec_to_se_field!(dst, dst_vec)
    return dst
end

end
