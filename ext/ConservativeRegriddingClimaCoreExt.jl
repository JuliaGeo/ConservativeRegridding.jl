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
    threaded = true,
    kwargs...
) where {M <: GOCore.Manifold}
    # Convert the SE source and FV destination grids into trees
    se_tree = Trees.treeify(manifold, src)
    fv_tree = Trees.treeify(manifold, dst)

    node_positions = se_node_positions(src)
    unique_wj = _unique_node_weights(src)
    per_elem_weights = se_node_weights(src)

    # Get the number of SE nodes and FV cells
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(src))
    Nh = Meshes.nelements(Topologies.mesh(Spaces.topology(src)))
    N_nodes = Nq^2 * Nh
    N_fv = prod(Trees.ncells(fv_tree))

    # Get all source SE nodes that lie in the FV cells we're remapping to
    candidate_pairs = _get_candidate_pairs(manifold, se_tree, fv_tree, threaded)
    elem_to_cells = Dict{Int, Vector{Int}}()
    for (se_elem_idx, cell_idx) in candidate_pairs
        push!(get!(elem_to_cells, se_elem_idx, Int[]), cell_idx)
    end

    # The weights are the SE node weights for the SE nodes in the FV cells we're remapping to
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

    # Find all rows of the weight matrix that have zero sum (i.e. no SE nodes in the FV cell)
    # and add the fallback intersection-area-weighted element averages to those rows
    zero_rows = findall(iszero, vec(sum(weight_matrix; dims=2)))
    weight_matrix = _add_intersection_fallback!(
        weight_matrix, candidate_pairs, zero_rows,
        manifold, se_tree, fv_tree, per_elem_weights, Nq, N_fv, N_nodes
    )

    # Get the areas of the FV cells, which are used for normalization
    dst_areas = ConservativeRegridding.areas(manifold, dst, fv_tree)

    # Allocate temporary arrays for the regridding operation
    dst_temp  = zeros(N_fv)
    src_temp  = zeros(N_nodes)

    return ConservativeRegridding.SEtoFVRegridder(weight_matrix, dst_areas, dst_temp, src_temp)
end

# FV source → SE destination
function ConservativeRegridding.Regridder(
    manifold::M, dst::ClimaCore.Spaces.AbstractSpectralElementSpace, src;
    threaded = true,
    kwargs...
) where {M <: GOCore.Manifold}
    # Convert the SE source and FV destination grids into trees
    se_tree = Trees.treeify(manifold, dst)
    fv_tree = Trees.treeify(manifold, src)

    # Get the SE node positions as a flat vector
    node_positions = se_node_positions(dst)

    # Get the number of SE nodes and FV cells
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(dst))
    Nh = Meshes.nelements(Topologies.mesh(Spaces.topology(dst)))
    N_nodes = Nq^2 * Nh
    N_fv = prod(Trees.ncells(fv_tree))

    # Get all destination SE nodes that lie in the FV cells we're remapping from
    candidate_pairs = _get_candidate_pairs(manifold, se_tree, fv_tree, threaded)
    node_cell_pairs = _find_nodes_in_cells(candidate_pairs, fv_tree, node_positions, Nq)
    node_idxs = first.(node_cell_pairs)
    cell_idxs  = last.(node_cell_pairs)

    # The weights are 1 for all SE nodes in the FV cells we're remapping from
    weight_matrix = SparseArrays.sparse(
        node_idxs, cell_idxs, ones(length(node_idxs)), N_nodes, N_fv
    )

    # Allocate temporary arrays for the regridding operation
    dst_temp = zeros(N_nodes)
    src_temp = zeros(N_fv)

    return ConservativeRegridding.FVtoSERegridder(weight_matrix, dst_temp, src_temp)
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
"""
function ConservativeRegridding.regrid!(
    dst::ClimaCore.Fields.Field,
    regridder::ConservativeRegridding.FVtoSERegridder,
    src::AbstractVector,
)
    dst_vec = regridder.dst_temp
    ConservativeRegridding.regrid!(dst_vec, regridder, src)
    vec_to_se_field!(dst, dst_vec)
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
