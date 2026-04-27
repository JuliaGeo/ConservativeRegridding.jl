module ConservativeRegriddingClimaCoreExt

import ConservativeRegridding
using ConservativeRegridding: Trees

import GeometryOpsCore as GOCore
import GeometryOps as GO

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

"""
    edge_representative(edge, ne) -> (i, j)

Pick a representative element on the given cube edge of a face that is
*interior* to the edge (i.e. not at a face corner). Used by
[`build_cube_connectivity`](@ref) so that the queried element-local face is
unambiguously the cube-edge crossing.
"""
function edge_representative(edge::Integer, ne::Integer)
    s = max(ne ÷ 2, 1)
    if edge == 1        # south
        return (s, 1)
    elseif edge == 2    # east
        return (ne, s)
    elseif edge == 3    # north
        return (s, ne)
    else                # west
        return (1, s)
    end
end

"""
    build_cube_connectivity(topology, ne) -> Trees.CubeFaceConnectivity

Build the cube-face connectivity table used by `neighbours` on a
`CubedSphereToplevelTree`. For each `(face F, edge edge_id)`, query
`Topologies.opposing_face` on a representative element interior to that edge,
and decode the result into `(neighbour_face, neighbour_edge, reversed)`.

This relies on two ClimaCore conventions that hold for `Topology2D` on the
default cubed-sphere mesh:

- `topology.elemorder == CartesianIndices((ne, ne, 6))`, so the global element
  index is `i + (j-1)*ne + (F-1)*ne²` (the same scheme used by
  `IndexOffsetQuadtreeCursor`).
- Element-local face IDs (1=south, 2=east, 3=north, 4=west) match our cube-edge
  IDs.
"""
function build_cube_connectivity(topology, ne::Integer)
    table = Array{Tuple{Int8, Int8, Bool}, 2}(undef, 4, 6)
    ne2 = ne * ne
    for F in 1:6, edge in 1:4
        i, j = edge_representative(edge, ne)
        elem = i + (j - 1) * ne + (F - 1) * ne2
        opelem, opface, reversed = Topologies.opposing_face(topology, elem, edge)
        F_prime = ((opelem - 1) ÷ ne2) + 1
        table[edge, F] = (Int8(F_prime), Int8(opface), Bool(reversed))
    end
    return Trees.CubeFaceConnectivity(table, Int(ne))
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

    connectivity = build_cube_connectivity(topology, ne)
    return Trees.CubedSphereToplevelTree(quadtrees, connectivity)
end

Trees.treeify(manifold::GOCore.Spherical, space::ClimaCore.Spaces.AbstractSpectralElementSpace) = Trees.treeify(manifold, Spaces.topology(space))

GOCore.best_manifold(mesh::Meshes.AbstractCubedSphere) = GOCore.Spherical(; radius = mesh.domain.radius)
GOCore.best_manifold(topology::Topologies.Topology2D) = GOCore.best_manifold(topology.mesh)
GOCore.best_manifold(space::ClimaCore.Spaces.AbstractSpectralElementSpace) = GOCore.best_manifold(Spaces.topology(space))

GOCore.best_manifold(field::ClimaCore.Fields.Field) = GOCore.best_manifold(getfield(field, :space))
Trees.treeify(manifold::GOCore.Spherical, field::ClimaCore.Fields.Field) = Trees.treeify(manifold, getfield(field, :space))








## Utility functions for getting values from ClimaCore fields
## Might be useful for when we implement regrid!

### Helper functions to interface with ClimaCore.jl
"""
    get_element_vertices(space::SpectralElementSpace2D)

Returns a vector of vectors, each containing the coordinates of the vertices
of an element. The vertices are in clockwise order for each element, and the
first coordinate pair is repeated at the end.

Also performs a check for zero area polygons, and throws an error if any are found.

This is the format expected by ConservativeRegridding.jl to construct a
Regridder object.
"""
function get_element_vertices(space)
    # Get the indices of the vertices of the elements, in clockwise order for each element
    topology = Spaces.topology(space)
    mesh = Topologies.mesh(topology)
    Nh = Meshes.nelements(mesh)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    vertex_inds = [
        CartesianIndex(i, j, 1, 1, e) # f and v are 1 for SpectralElementSpace2D
        for e in 1:Nh
        for (i, j) in [(1, 1), (1, Nq), (Nq, Nq), (Nq, 1), (1, 1)]
    ] # repeat the first coordinate pair at the end

    # Get the lat and lon at each vertex index
    coords = Fields.coordinate_field(space)
    vertex_coords = [
        (Fields.field_values(coords.lat)[ind], Fields.field_values(coords.long)[ind])
        for ind in vertex_inds
    ]

    # Put each polygon into a vector, with the first coordinate pair repeated at the end
    vertices = collect(Iterators.partition(vertex_coords, 5))

    # Check for zero area polygons
    for polygon in vertices
        if allequal(first.(polygon)) || allequal(last.(polygon))
            @error "Zero area polygon found in vertices" polygon
        end
    end
    return vertices
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
    integrate_each_element(field)

Integrate the field over each element of the space.
Returns a vector with length equal to the number of elements in the space,
containing the integrated value over the nodes of each element.
"""
function integrate_each_element(field)
    space = axes(field)
    topology = Spaces.topology(space)
    mesh = Topologies.mesh(topology)

    weighted_values =
        RecursiveApply.rmul.(
            Spaces.weighted_jacobian(space),
            Fields.todata(field),
        )

    Nh = Meshes.nelements(mesh)
    integral_each_element = zeros(Float64, Nh)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))

    # Sum over the nodes of each element to get the integral of the field over each element
    integral_each_element = vec(sum(parent(weighted_values); dims=(1, 2)))

    return integral_each_element
end

"""
    get_value_per_element!(value_per_element, field, ones_field)

Get one value per element of a field by integrating over the nodes of
each element and dividing by the area of the element. The result is stored in
`value_per_element`, which is expected to be a Float-valued vector of length equal
to the number of elements in the space.

Here `ones_field` is a field on the same space as `field` with all
values set to 1.
"""
function get_value_per_element!(value_per_element, field, ones_field)
    integral_each_element = integrate_each_element(field)
    area_each_element = integrate_each_element(ones_field)
    value_per_element .= integral_each_element ./ area_each_element
    return nothing
end

"""
    set_value_per_element!(field, value_per_element)

Set the values of a field with the provided values in each element.
Each node within an element will have the same value.

The input vector is expected to be of length equal to the number of elements in
the space.
"""
function set_value_per_element!(field, value_per_element)
    space = axes(field)
    topology = Spaces.topology(space)
    mesh = Topologies.mesh(topology)
    Nh = Meshes.nelements(mesh)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    @assert length(value_per_element) == Nh "Length of value_per_element must be equal to the number of elements in the space"

    # Set all nodes in each element to the value per element
    for i in 1:Nq, j in 1:Nq
        set_datalayout!(Fields.field_values(field), i, j, value_per_element)
    end

    return field
end

"""
    set_datalayout!(values::DataLayouts.IJFH, i, j, value_per_element)
    set_datalayout!(values::DataLayouts.VIJFH, i, j, value_per_element)

Set the values of the provided data laout with the given values in each element.
The input vector is expected to be of length equal to the number of elements in
the space.

`i` and `j` are the indices of the element to set the values of. All nodes
in that element will be set to the same value.

We need two methods of this function because the data layout may have a vertical
dimension (VIJFH) or not (IJFH). This is true even though we only regrid 2D fields,
as a 2D field constructed by taking a level of a 3D field will have a vertical dimension.
"""
function set_datalayout!(values::DataLayouts.IJFH, i, j, value_per_element)
    view(parent(values), i, j, 1, :) .= value_per_element
end
function set_datalayout!(values::DataLayouts.VIJFH, i, j, value_per_element)
    view(parent(values), i, j, 1, 1, :) .= value_per_element
end

end
