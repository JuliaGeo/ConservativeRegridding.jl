module ConservativeRegriddingClimaCoreExt

import ConservativeRegridding
using ConservativeRegridding: Trees

import GeometryOpsCore as GOCore
import GeometryOps as GO
import GeoInterface as GI

using GeometryOps.UnitSpherical: UnitSphericalPoint
using LinearAlgebra: normalize
using StaticArrays: SVector, StaticArrays

using ClimaCore:
    CommonSpaces, Fields, Spaces, RecursiveApply, Meshes, Quadratures, Topologies, DataLayouts, ClimaComms
import ClimaCore
import ClimaCore.Utilities: linear_ind

"""
    coords_for_face(mesh::CubedSphereMesh, face_idx)::Matrix{UnitSphericalPoint}

Normalized vertex coordinates for a cubed-sphere face: an `(ne+1)×(ne+1)` matrix of points.
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
    flat_nodal_data(data::DataLayouts.AbstractData) → Vector

Flatten nodal storage to a `Vector` via [`DataLayouts.data2array`](@ref) (memory layout:
for `IJFH`, `i` fastest, then `j`, then horizontal indices). For layouts with a vertical
dimension (`VIJFH`, etc.) `data2array` is an ``N_v \\times N_h`` matrix; the first vertical
level is taken. Unlike `Fields.field2array`, accepts raw `AbstractData` (e.g.
`Fields.field_values(coords.lat)`, `Spaces.weighted_jacobian(space)`).
"""
function flat_nodal_data(data::DataLayouts.AbstractData)
    array = DataLayouts.data2array(data)
    if array isa AbstractVector
        return vec(array)
    elseif ndims(array) == 2
        return vec(view(array, 1, :))
    else
        error("Unexpected data2array shape: $(size(array))")
    end
end

"""
    se_node_positions(space) → Vector{UnitSphericalPoint}

Positions of all SE nodes as a flat vector. Ordering: all Nq² nodes of element 1
(i fastest), then element 2, etc.
"""
function se_node_positions(space)
    coords = Fields.coordinate_field(space)
    lat_flat  = flat_nodal_data(Fields.field_values(coords.lat))
    long_flat = flat_nodal_data(Fields.field_values(coords.long))
    transform = GO.UnitSphereFromGeographic()
    return [transform((long_flat[k], lat_flat[k])) for k in eachindex(lat_flat)]
end

"""
    se_node_weights(space) → Vector{Float64}

Jacobian integration weights ``W_{e,i,j}`` for all SE nodes as a flat vector.
Same ordering as [`se_node_positions`](@ref).
"""
function se_node_weights(space)
    wj = Spaces.weighted_jacobian(space)
    return flat_nodal_data(wj)
end

# ────────────────────────────────────────────────────────────────────────────
# Inverse element map for the equiangular cubed sphere
# ────────────────────────────────────────────────────────────────────────────

"""
    element_face_local_indices(topology, elem_idx) -> (face, ie, je)

Cubed-sphere face (1–6) and local 2D indices `(ie, je)` for global element `elem_idx`.
Reads `topology.elemorder` (face-major `CartesianIndices((ne, ne, 6))` or a space-filling
permutation `Vector{CartesianIndex{3}}`), so it is correct for both regular and
Gilbert-ordered topologies.
"""
function element_face_local_indices(topology::Topologies.Topology2D, elem_idx::Int)
    ci = topology.elemorder[elem_idx]
    return ci[3], ci[1], ci[2]   # (face, ie, je)
end

"""
    inverse_element_map(space, elem_idx, x) -> (ξ, η)

Element-local reference coordinates `(ξ, η) ∈ [-1, 1]²` of a 3D sphere point `x` known to
lie inside element `elem_idx`. Delegates to `ClimaCore.Meshes.reference_coordinates`, which
handles both `IntrinsicMap` (closed-form equiangular inversion) and `NormalizedBilinearMap`
(bilinear-invert against the four corners; the cubed-sphere default). The two maps agree at
corner GLL nodes but differ at interior nodes, so a hand-rolled equiangular-only inverse
fails for the default mesh.
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

SE element Jacobian at `(ξ, η)` via Lagrange interpolation of nodal weighted-Jacobian
values on `space` (PDF Eq. 47):

    Jᵉ(ξ, η) ≈ Σ_{p,q} Jᵉₚᵩ ϕₚ(ξ) ϕᵩ(η)

where `Jᵉₚᵩ = WJ[p, q, 1, elem_idx] / (wₚ wᵩ)` is the unweighted Jacobian recovered from
`Spaces.weighted_jacobian` storage.
"""
function element_jacobian_at(space::ClimaCore.Spaces.AbstractSpectralElementSpace,
                             elem_idx::Int, ξ, η)
    qs = Spaces.quadrature_style(space)
    ξs, ws = Quadratures.quadrature_points(Float64, qs)
    Nq = length(ξs)

    WJ = parent(Spaces.weighted_jacobian(space))
    Mξ = Quadratures.interpolation_matrix(SVector(ξ), ξs)
    Mη = Quadratures.interpolation_matrix(SVector(η), ξs)

    Jᵉ = 0.0
    @inbounds for q in 1:Nq, p in 1:Nq
        Jₚᵩ = WJ[p, q, 1, elem_idx] / (ws[p] * ws[q])
        Jᵉ += Jₚᵩ * Mξ[1, p] * Mη[1, q]
    end
    return Jᵉ
end

# ────────────────────────────────────────────────────────────────────────────
# Principled B-accumulator (Task 6, PDF Eq. 48)
# ────────────────────────────────────────────────────────────────────────────

"""
    accumulate_principled_b(manifold, space, elem_idx, intersection_polygon;
                            triangle_quad_degree) -> Matrix{Float64}

Principled `B(k, (e, i, j))` weights for one source SE element `e = elem_idx` and one
destination physical-space polygon `intersection_polygon`. Returns an `Nq × Nq` matrix with

    B[i, j] ≈ ∫_{intersection_polygon} ϕᵢ(ξ) ϕⱼ(η) dA_phys       (PDF Eq. 48)

The Jacobian factor in PDF Eq. 18 cancels under change of variables:
`∫_{ref} ϕᵢϕⱼ Jᵉ dξ dη = ∫_{phys} ϕᵢ ϕⱼ dA`. Approach: fan-triangulate the polygon from its
centroid, apply a barycentric Gauss rule per triangle, and evaluate the Lagrange basis at
each quadrature point (`inverse_element_map` gives `(ξ, η)`).
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
            Mξ = Quadratures.interpolation_matrix(SVector(ξ), ξs)
            Mη = Quadratures.interpolation_matrix(SVector(η), ξs)

            # Reference rule's weights sum to 1/2 (ref triangle area); to map
            # onto a physical triangle of area Aₜ, scale by Aₜ/(1/2) = 2 Aₜ.
            wAᵧ = wᵧ * 2 * Aₜ

            @inbounds for j in 1:Nq, i in 1:Nq
                B[i, j] += wAᵧ * Mξ[1, i] * Mη[1, j]
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
    poly = GI.Polygon(StaticArrays.SA[GI.LinearRing(StaticArrays.SA[q₁, q₂, q₃, q₁])])
    # `::Float64` barrier: GO.area on a spherical polygon infers `Any`, which
    # otherwise boxes every quadrature weight downstream in accumulate_principled_b.
    return GO.area(GO.Spherical(), poly)::Float64
end

function spherical_triangle_area(::GOCore.Planar, p₁, p₂, p₃)
    return ConservativeRegridding.TriangleQuadrature.planar_triangle_area(
        ((p₁[1], p₁[2]), (p₂[1], p₂[2]), (p₃[1], p₃[2]))
    )
end

"""
    se_field_to_vec(field)

Flat vector of a ClimaCore field's nodal values. Same ordering as [`se_node_positions`](@ref).
"""
function se_field_to_vec(field)
    return flat_nodal_data(Fields.field_values(field))
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

## Regridder constructors

const SESpaceOrField = Union{ClimaCore.Spaces.AbstractSpectralElementSpace, ClimaCore.Fields.Field}

se_space(space::ClimaCore.Spaces.AbstractSpectralElementSpace) = space
se_space(field::ClimaCore.Fields.Field) = axes(field)

# SE source → FV destination (principled polygon-intersection, PDF Appendix A)
function ConservativeRegridding.Regridder(
    manifold::M, dst, src::SESpaceOrField;
    triangle_quad_degree::Union{Int, Nothing} = nothing,
    threaded = true,
    kwargs...
) where {M <: GOCore.Manifold}
    return se_to_fv_principled(manifold, dst, se_space(src); threaded, triangle_quad_degree, kwargs...)
end

# SE → FV operator (`InPlace`, principled polygon-intersection, PDF Appendix A):
# per (elem, cell) pair, push an Nq² block of B-weights per intersection polygon.
struct SEToFVIntersectionOperator{M <: GOCore.Manifold, SP}
    manifold::M
    src_space::SP            # SE source space
    Nq::Int
    triangle_quad_degree::Int
    N_fv::Int
    N_nodes::Int
end

ConservativeRegridding.IntersectionReturnStyle(::SEToFVIntersectionOperator) = ConservativeRegridding.InPlace()
ConservativeRegridding.output_matrix_size(op::SEToFVIntersectionOperator, ::Any, ::Any) = (op.N_fv, op.N_nodes)

function (op::SEToFVIntersectionOperator)(rows, cols, vals, (elem, cell), src_tree, dst_tree)
    elem_poly = Trees.getcell(src_tree, elem)
    cell_poly = Trees.getcell(dst_tree, cell)

    intersection_result = if op.manifold isa GOCore.Spherical
        GO.intersection(GO.ConvexConvexSutherlandHodgman(op.manifold),
                        elem_poly, cell_poly; target = GI.PolygonTrait())
    else
        GO.intersection(GO.FosterHormannClipping(GO.Planar()),
                        elem_poly, cell_poly; target = GI.PolygonTrait())
    end

    # GO.intersection may return a single Polygon, a Vector{Polygon}, or
    # nothing for an empty intersection. Normalize to an iterable.
    intersection_polys = if isnothing(intersection_result)
        ()
    elseif intersection_result isa AbstractVector
        intersection_result
    else
        (intersection_result,)   # single Polygon
    end

    for ipoly in intersection_polys
        B = accumulate_principled_b(op.manifold, op.src_space, elem, ipoly;
                                    triangle_quad_degree = op.triangle_quad_degree)
        offset = (elem - 1) * op.Nq^2
        for j in 1:op.Nq, i in 1:op.Nq
            Bᵢⱼ = B[i, j]
            Bᵢⱼ == 0 && continue
            push!(rows, cell)
            push!(cols, offset + linear_ind((op.Nq, op.Nq), (i, j)))
            push!(vals, Bᵢⱼ)
        end
    end
    return nothing
end

function se_to_fv_principled(manifold, dst, src; threaded, triangle_quad_degree, kwargs...)
    se_tree = Trees.treeify(manifold, src)
    fv_tree = Trees.treeify(manifold, dst)

    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(src))
    Nh = Meshes.nelements(Topologies.mesh(Spaces.topology(src)))
    N_nodes = Nq^2 * Nh
    N_fv = prod(Trees.ncells(fv_tree))

    triangle_quad_degree = something(triangle_quad_degree, 2 * (Nq - 1))

    # Assemble the N_fv × N_nodes matrix via the shared parallel driver: src = SE
    # elements, dst = FV cells, so candidate pairs are uniformly (elem, cell).
    op = SEToFVIntersectionOperator(manifold, src, Nq, triangle_quad_degree, N_fv, N_nodes)
    # Extra kwargs (e.g. `normalize`) are absorbed by the signature and not forwarded;
    # `intersection_areas` only accepts assembly kwargs.
    intersections = ConservativeRegridding.intersection_areas(
        manifold, GOCore.booltype(threaded), fv_tree, se_tree;
        intersection_operator = op,
    )

    dst_areas = ConservativeRegridding.areas(manifold, dst, fv_tree)
    src_areas = Vector{eltype(dst_areas)}(se_node_weights(src))

    return ConservativeRegridding.Regridder(
        intersections, dst_areas, src_areas,
        zeros(N_fv), zeros(N_nodes),
    )
end

# FV source → SE destination (per-element L2 projection)
function ConservativeRegridding.Regridder(
    manifold::M, dst::SESpaceOrField, src;
    triangle_quad_degree::Union{Int, Nothing} = nothing,
    threaded = true,
    kwargs...
) where {M <: GOCore.Manifold}
    return fv_to_se_l2_projection(manifold, se_space(dst), src; threaded, triangle_quad_degree, kwargs...)
end

# Disambiguate SE → SE: no longer supported on this branch.
function ConservativeRegridding.Regridder(
    manifold::M,
    dst::SESpaceOrField,
    src::SESpaceOrField;
    kwargs...
) where {M <: GOCore.Manifold}
    error("SE → SE regridding is not supported. Use SE → FV → SE via two regridders.")
end

"""
    compute_local_mass_matrix(space, elem_idx) -> Matrix{Float64}

Dense `Nq² × Nq²` mass matrix for SE element `elem_idx`,

    M^{e}_{(a,b),(i,j)} = ∫_{e} ϕₐ(ξ) ϕᵦ(η) ϕᵢ(ξ) ϕⱼ(η) Jᵉ(ξ,η) dξ dη ,

row/col flattening per `ClimaCore.Utilities.linear_ind()`. The integrand has bidegree
`(2(Nq-1), 2(Nq-1))` (basis-function product) plus `(Nq-1, Nq-1)` (Lagrange-interpolated
`Jᵉ`), total `(3(Nq-1), 3(Nq-1))`. A GLL rule with `n` points/direction is exact to degree
`2n - 1`, so `n = ceil((3Nq - 2)/2) + 1`.
"""
function compute_local_mass_matrix(
    space::ClimaCore.Spaces.AbstractSpectralElementSpace, elem_idx::Int,
)
    qs = Spaces.quadrature_style(space)
    ξs_se, _ = Quadratures.quadrature_points(Float64, qs)
    Nq = length(ξs_se)

    # GLL with n = (3Nq - 2) / 2 points exactly integrates
    n = ceil(Int, (3Nq - 2) / 2) + 1
    qs_q = Quadratures.GLL{n}()
    ξs_q, ws_q = Quadratures.quadrature_points(Float64, qs_q)
    Nq_q = length(ξs_q)

    Nq² = Nq * Nq
    M = zeros(Nq², Nq²)

    @inbounds for q in 1:Nq_q, p in 1:Nq_q
        ξ = ξs_q[p]
        η = ξs_q[q]
        wξ = ws_q[p]
        wη = ws_q[q]

        Mξ = Quadratures.interpolation_matrix(SVector(ξ), ξs_se)
        Mη = Quadratures.interpolation_matrix(SVector(η), ξs_se)
        Jᵉ = element_jacobian_at(space, elem_idx, ξ, η)

        wJ = wξ * wη * Jᵉ
        for jb in 1:Nq, ja in 1:Nq
            row = linear_ind((Nq, Nq), (ja, jb))
            ϕaξ = Mξ[1, ja]
            ϕbη = Mη[1, jb]
            for jj in 1:Nq, ii in 1:Nq
                col = linear_ind((Nq, Nq), (ii, jj))
                M[row, col] += wJ * ϕaξ * Mξ[1, ii] * ϕbη * Mη[1, jj]
            end
        end
    end
    return M
end

# FV → SE operator (`InPlace`, per-element L2 projection). The per-element mass-matrix
# solve needs all of an element's cells at once, so `work_items` makes the element
# (not the candidate pair) the unit of work. Assembled matrix is N_nodes × N_fv.
struct FVToSEIntersectionOperator{M <: GOCore.Manifold, SP}
    manifold::M
    dst_space::SP            # SE destination space
    Nq::Int
    triangle_quad_degree::Int
    Nh::Int
    N_nodes::Int
    N_fv::Int
end

ConservativeRegridding.IntersectionReturnStyle(::FVToSEIntersectionOperator) = ConservativeRegridding.InPlace()
ConservativeRegridding.output_matrix_size(op::FVToSEIntersectionOperator, ::Any, ::Any) = (op.N_nodes, op.N_fv)

# Group (elem, cell) pairs into one (elem, cells) item per non-empty element, preserving
# candidate-pair order so serial assembly matches the old per-element loop.
function ConservativeRegridding.work_items(op::FVToSEIntersectionOperator, candidate_pairs)
    cells_by_elem = [Int[] for _ in 1:op.Nh]
    for (elem, cell) in candidate_pairs
        push!(cells_by_elem[elem], cell)
    end
    return [(e, cells_by_elem[e]) for e in 1:op.Nh if !isempty(cells_by_elem[e])]
end

function (op::FVToSEIntersectionOperator)(rows, cols, vals, (elem, cells), src_tree, dst_tree)
    Nq = op.Nq
    elem_poly = Trees.getcell(src_tree, elem)

    # Phase 1: accumulate B for this element, keyed by destination cell, value = Nq² vector.
    d = Dict{Int, Vector{Float64}}()
    for cell in cells
        cell_poly = Trees.getcell(dst_tree, cell)

        intersection_result = if op.manifold isa GOCore.Spherical
            GO.intersection(GO.ConvexConvexSutherlandHodgman(op.manifold),
                            elem_poly, cell_poly; target = GI.PolygonTrait())
        else
            GO.intersection(GO.FosterHormannClipping(GO.Planar()),
                            elem_poly, cell_poly; target = GI.PolygonTrait())
        end

        intersection_polys = if isnothing(intersection_result)
            ()
        elseif intersection_result isa AbstractVector
            intersection_result
        else
            (intersection_result,)
        end

        for ipoly in intersection_polys
            B = accumulate_principled_b(op.manifold, op.dst_space, elem, ipoly;
                                        triangle_quad_degree = op.triangle_quad_degree)
            v = get!(d, cell) do
                zeros(Nq^2)
            end
            for jb in 1:Nq, ja in 1:Nq
                Bᵢⱼ = B[ja, jb]
                Bᵢⱼ == 0 && continue
                v[linear_ind((Nq, Nq), (ja, jb))] += Bᵢⱼ
            end
        end
    end
    isempty(d) && return nothing

    # Phase 2: solve Mᵉ f_dst = b for each cell column, push the COO block.
    #
    # Mᵉ comes from tensor-product GLL on the reference square but B from fan
    # triangulation on the great-circle physical polygon, so the two
    # quadratures integrate over slightly different domains. We rescale each
    # row of Mᵉ so that its row sum equals the column-sum of B for that
    # element — what Mᵉ·1 must equal by partition of unity if both sides used
    # identical quadrature. This forces exact constant preservation.
    Mᵉ = compute_local_mass_matrix(op.dst_space, elem)

    b_full = zeros(Nq^2)
    for b_vec in values(d)
        b_full .+= b_vec
    end

    # If b_full == 0 it means we are hitting regions with no
    # overlap (for example regridding a tripolar grid that ends
    # at 80ᵒ S onto a SE grid that covers the sphere).
    # We cover for those cases.
    covered = findall(!=(0), b_full)
    isempty(covered) && return nothing
    Mᶜ = Mᵉ[covered, covered]

    row_sums = vec(sum(Mᶜ; dims=2))
    for (rc, r) in enumerate(covered)
        row_sums[rc] == 0 && continue
        scale = b_full[r] / row_sums[rc]
        @inbounds for c in axes(Mᶜ, 2)
            Mᶜ[rc, c] *= scale
        end
    end

    Mᶜ_inv = inv(Mᶜ)

    offset = (elem - 1) * Nq^2
    for (cell, b_vec) in d
        new_col = Mᶜ_inv * b_vec[covered]
        for (rc, r) in enumerate(covered)
            val = new_col[rc]
            val == 0 && continue
            push!(rows, offset + r)
            push!(cols, cell)
            push!(vals, val)
        end
    end
    return nothing
end

"""
    fv_to_se_l2_projection(manifold, dst, src; ...)

Per-element L2 projection FV → SE. Same `B` accumulation as the principled path, but
per-element rows are multiplied by the *full* mass-matrix inverse `(M^{e})^{-1}` rather than
divided by the lumped diagonal `Wᵉᵢⱼ` (cf. PDF Eq. 30). This preserves constants exactly and
is higher-order accurate: `f_dst = (M^{e})^{-1} (Bᵀ f_src)|_e` is the optimal L2 fit on the
SE basis over element `e`.
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

    # Assemble the N_nodes × N_fv matrix via the shared parallel driver: src = SE
    # elements, dst = FV cells, with `work_items` regrouping pairs by element so the
    # per-element mass-matrix solve sees all of an element's cells at once.
    op = FVToSEIntersectionOperator(manifold, dst, Nq, triangle_quad_degree, Nh, N_nodes, N_fv)
    # Extra kwargs (e.g. `normalize`) are absorbed by the signature and not forwarded;
    # `intersection_areas` only accepts assembly kwargs.
    intersections = ConservativeRegridding.intersection_areas(
        manifold, GOCore.booltype(threaded), fv_tree, se_tree;
        intersection_operator = op,
    )

    src_areas = ConservativeRegridding.areas(manifold, src, fv_tree)
    # inv-mass already baked into `intersections`; ones make pipeline normalize a no-op
    dst_areas = ones(N_nodes)

    return ConservativeRegridding.Regridder(
        intersections, dst_areas, src_areas,
        zeros(N_nodes), zeros(N_fv),
    )
end

## Pipeline overrides for ClimaCore Fields
#
# A `ClimaCore.Fields.Field` is the marker for SE-side data. Source-side: flatten
# nodal values into the regridder's work buffer during `initialize_regridding!`.
# Destination-side: copy the work buffer back into the field and apply weighted
# DSS in `finalize_regridding!`. The FV→SE matrix already has the inverse mass
# baked in, so the standard normalize step is skipped here.

ConservativeRegridding.extract_source_arraylike(src::ClimaCore.Fields.Field, regridder; kwargs...) = regridder.src_temp
ConservativeRegridding.extract_dest_arraylike(dst::ClimaCore.Fields.Field, regridder; kwargs...) = regridder.dst_temp

function ConservativeRegridding.initialize_regridding!(
    regridder, src::ClimaCore.Fields.Field, src_arraylike::AbstractVector; kwargs...,
)
    src_arraylike .= se_field_to_vec(src)
    return regridder
end

function ConservativeRegridding.finalize_regridding!(
    dst::ClimaCore.Fields.Field, regridder, dst_arraylike::AbstractVector; kwargs...,
)
    vec_to_se_field!(dst, dst_arraylike)
    Spaces.weighted_dss!(dst)
    return dst
end

end
