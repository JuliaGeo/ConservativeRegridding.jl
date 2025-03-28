
regridder(grid1, grid2) = GeometryOps.area_of_intersection_operator(grid1, grid2)


regridder!(regridder::AbstractMatrix, grid1, grid2) = GeometryOps.area_of_intersection_operator!(regridder, grid1, grid2)

"""$(TYPEDSIGNATURES)
Returns area vectors (out, in) for the grids used to create the regridder.
The area vectors are computed by summing the regridder along the first and second dimensions
as the regridder is a matrix of the intersection areas between each grid cell between the
two grids."""
areas(regridder::AbstractMatrix) = area(regridder, dims=:out), area(regridder, dims=:in)

"""$(TYPEDSIGNATURES) Area vector from `regridder`, `dims` can be `:in` or `:out`."""
area(regridder::AbstractMatrix; dims) = area(regridder, dims)                     # pass on keyword as positional argument for dispatch

# "in" is a sum along the 2nd dimension of the matrix, returning a vector of length of the 1st dimension (the output grid)
area(regridder::AbstractMatrix, dims::Val{:in}) = vec(sum(regridder, dims=2))        
area(regridder::AbstractMatrix, dims::Val{:out}) = vec(sum(regridder, dims=1))     # "out" vice versa