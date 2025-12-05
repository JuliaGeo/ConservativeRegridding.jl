########################################################
# This test uses ConservativeRegridding.jl to regrid between two ClimaCore spaces.
#
# Note that ClimaCore.jl uses spectral element spaces, which are not yet supported
# in ConservativeRegridding.jl. To use the existing functionality, we store a single
# value per element on the field, rather than one value per node. In this way the
# spectral element space can be regridded as a finite volume space would be.
#
# This is an inaccurate approximation that we'll want to improve in the future,
# but it's okay to get things started.
########################################################


using ClimaCore:
    CommonSpaces, Fields, Spaces, RecursiveApply, Meshes, Quadratures
using ConservativeRegridding
using Statistics
using Test

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


### Test regridding between two ClimaCore grids
space1 = CommonSpaces.CubedSphereSpace(;
    radius = 10,
    n_quad_points = 3,
    h_elem = 8,
)
space2 = CommonSpaces.CubedSphereSpace(;
    radius = 10,
    n_quad_points = 4,
    h_elem = 6,
)

vertices1 = get_element_vertices(space1)
vertices2 = get_element_vertices(space2)

# Pass in destination vertices first, source vertices second
regridder_1_to_2 = ConservativeRegridding.Regridder(vertices2, vertices1)
regridder_2_to_1 = ConservativeRegridding.Regridder(vertices1, vertices2)

# Define a field on the first space, to use as our source field
field1 = Fields.coordinate_field(space1).lat
ones_field1 = Fields.ones(space1)

# Get one value per element in the source field, equal to the quadrature-weighted average of the
# values at nodes of the element
value_per_element1 = zeros(Float64, Meshes.nelements(space1.grid.topology.mesh))
get_value_per_element!(value_per_element1, field1, ones_field1)

# Allocate a vector with length equal to the number of elements in the target space
value_per_element2 = zeros(Float64, Meshes.nelements(space2.grid.topology.mesh))
ConservativeRegridding.regrid!(value_per_element2, regridder_1_to_2, value_per_element1)

# Now that we have our regridded vector, put it onto a field on the second space
field2 = Fields.zeros(space2)
set_value_per_element!(field2, value_per_element2)
field1_one_value_per_element = Fields.zeros(space1)
set_value_per_element!(field1_one_value_per_element, value_per_element1)

# Test our helper functions
# Check that integrating over each element and summing gives the same result as integrating over the whole domain
@test isapprox(sum(integrate_each_element(field1)), sum(field1), atol = 1e-12)
# Check that integrating 1 over each element and summing gives the same result as integrating 1 over the whole domain
@test sum(integrate_each_element(ones_field1)) ≈ sum(ones_field1)

# Check the error of converting to one value per element
abs_error_one_value_per_element = abs(sum(field1_one_value_per_element) - sum(field1))
@test abs_error_one_value_per_element < 2e-11
@test isapprox(mean(field1), mean(field1_one_value_per_element), atol=1e-14)

# Check the global conservation error of the overall regridding
abs_error = abs(sum(field1) - sum(field2))
@test abs_error < 1e-12
@test isapprox(mean(field1), mean(field2), atol=1e-15)

# # Plot the fields for visual comparison
# using ClimaCoreMakie
# using GLMakie
# fig = ClimaCoreMakie.fieldheatmap(field1)
# save("field1.png", fig)
# fig = ClimaCoreMakie.fieldheatmap(field1_one_value_per_element)
# save("field1_one_value_per_element.png", fig)
# fig = ClimaCoreMakie.fieldheatmap(field2)
# save("field2.png", fig)
