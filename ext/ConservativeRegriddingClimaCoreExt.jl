module ConservativeRegriddingClimaCoreExt

import ConservativeRegridding
using ConservativeRegridding: Trees

import GeometryOpsCore as GOCore
import GeometryOps as GO

using GeometryOps.UnitSpherical: UnitSphericalPoint
using LinearAlgebra: normalize

using ClimaCore:
    CommonSpaces, Fields, Spaces, RecursiveApply, Meshes, Quadratures, Topologies
import ClimaCore

"""
    coords_for_face(mesh::CubedSphereMesh, face_idx)::Matrix{UnitSphericalPoint}

Get the right coordinates for a face of a cubed sphere.
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

function Trees.treeify(manifold::GOCore.Spherical, mesh::Meshes.AbstractCubedSphere)
    ne = mesh.ne
    # There are always 6 faces of a cubed sphere - 
    # see the ClimaCore docs on AbstractCubedSphere
    # for an explanation on how they are connected.
    # TODO: set the border elements of each cubed sphere face
    # explicitly equal to each other, so that there are no numerical 
    # inaccuracies.
    face_idxs = 1:6

    # all_coords = map(i -> coords_for_face(mesh, i), face_idxs)
    # Create a quadtree for each face.
    quadtrees = map(face_idxs) do face_idx
        coords = coords_for_face(mesh, face_idx)
        Trees.FaceAwareQuadtreeCursor(
            Trees.CellBasedGrid(manifold, coords), 
            face_idx
        )
    end

    return Trees.CubedSphereToplevelTree(quadtrees)
end

Trees.treeify(manifold::GOCore.Spherical, space::ClimaCore.Spaces.AbstractSpectralElementSpace) = Trees.treeify(manifold, space.grid.topology.mesh)

GOCore.best_manifold(mesh::Meshes.AbstractCubedSphere) = GOCore.Spherical(; radius = mesh.domain.radius)
GOCore.best_manifold(space::ClimaCore.Spaces.AbstractSpectralElementSpace) = GOCore.best_manifold(space.grid.topology.mesh)










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
    Nh = Meshes.nelements(space.grid.topology.mesh)
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
    weighted_values =
        RecursiveApply.rmul.(
            Spaces.weighted_jacobian(space),
            Fields.todata(field),
        )

    Nh = Meshes.nelements(space.grid.topology.mesh)
    integral_each_element = zeros(Float64, Nh)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))
    for e in 1:Nh # loop over each element
        for i in 1:Nq
            for j in 1:Nq
                integral_each_element[e] += weighted_values[CartesianIndex(i, j, 1, 1, e)]
            end
        end
    end
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
    Nh = Meshes.nelements(space.grid.topology.mesh)
    Nq = Quadratures.degrees_of_freedom(Spaces.quadrature_style(space))

    @assert length(value_per_element) == Nh "Length of value_per_element must be equal to the number of elements in the space"

    # Set the value in each node of each element to the value per element
    for e in 1:Nh
        for i in 1:Nq
            for j in 1:Nq
                Fields.field_values(field)[CartesianIndex(i, j, 1, 1, e)] =
                    value_per_element[e]
            end
        end
    end
    return field
end



end