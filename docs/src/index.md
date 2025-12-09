# ConservativeRegridding.jl

ConservativeRegridding.jl provides functionality to regrid between two arbitrary grids.
A grid is a tessellation of a space into polygons (or grid cells), each with an associated value.
Data on a grid is referred to as a _field_, whereas the grid itself defines the tessellation.

Regridding is performed **conservatively**, meaning the area-weighted mean is preserved.
This is achieved by computing the intersection areas between all combinations of grid cells
from the source and destination grids. These intersection areas provide the weights for
averaging from neighboring cells.

## Quick Start: SpeedyWeather Grid Transfer

Here's an example of regridding between two different geodesic grids from [SpeedyWeather.jl](https://github.com/SpeedyWeather/SpeedyWeather.jl):

```julia
using SpeedyWeather
using RingGrids
using ConservativeRegridding

# Helper function to get polygon faces for SpeedyWeather grids
function get_faces(field)
    grid = field.grid
    Grid = typeof(grid)
    nlat_half = grid.nlat_half
    npoints = RingGrids.get_npoints2D(field)
    E, S, W, N = RingGrids.get_vertices(Grid, nlat_half)
    faces = Matrix{NTuple{2, Float64}}(undef, 5, npoints)
    @inbounds for ij in 1:npoints
        faces[1, ij] = (E[1, ij], E[2, ij])
        faces[2, ij] = (S[1, ij], S[2, ij])
        faces[3, ij] = (W[1, ij], W[2, ij])
        faces[4, ij] = (N[1, ij], N[2, ij])
        faces[5, ij] = (E[1, ij], E[2, ij])  # close the polygon
    end
    return faces
end

# Create random data on two different geodesic grid types
field1 = rand(OctaHEALPixGrid, 24)
field2 = rand(OctaminimalGaussianGrid, 24)

# Get the polygon faces (vertices) for each grid
faces1 = get_faces(field1)
faces2 = get_faces(field2)

# Build the regridder (internally converts vertices to polygons and fixes antimeridian crossings)
R = ConservativeRegridding.Regridder(faces1, faces2)

# Regrid from field2 to field1
ConservativeRegridding.regrid!(field1, R, field2)

# Regrid in the reverse direction using the transpose
ConservativeRegridding.regrid!(field2, transpose(R), field1)
```

The key advantage is that the `Regridder` only needs to be constructed once. After that,
both forward regridding (via `R`) and backward regridding (via `transpose(R)`) can be
performed efficiently without recomputing intersection areas.

## Mathematics of Regridding

### The Intersection Area Matrix

Conservative regridding is built around a matrix ``A`` of intersection areas between
source and destination grid cells. For a source grid with ``m`` cells and a destination
grid with ``n`` cells, the matrix ``A`` is ``n \times m``, where each entry ``A_{ij}``
represents the area of intersection between destination cell ``i`` and source cell ``j``.

The algorithm uses an efficient spatial tree structure (STRtree) to compute only the
non-zero intersections, avoiding the ``O(nm)`` cost of checking all cell pairs.

### Forward Regridding

Let ``s`` be a vector of field values on the source grid and ``d`` the destination field
values. The forward regrid operation computes:

```math
d_i = \frac{\sum_j A_{ij} s_j}{a^d_i}
```

or in matrix form:

```math
d = \frac{A s}{a^d}
```

where ``a^d_i`` is the area of destination cell ``i``, and the division is element-wise.
The matrix-vector product ``As`` yields values weighted by intersection areas, and the
division by ``a^d`` normalizes to obtain the regridded field values.

### Backward Regridding

The same intersection matrix ``A`` can be reused for backward regridding by transposing it:

```math
\tilde{s} = \frac{A^T d}{a^s}
```

where ``a^s_j`` is the area of source cell ``j``. The tilde on ``\tilde{s}`` emphasizes that
the round-trip operation ``s \to d \to \tilde{s}`` does not recover the original field exactly—conservative regridding preserves the mean but generally reduces variance.

### Conservation Property

Conservative regridding preserves the area-weighted mean:

```math
\frac{\sum_i d_i a^d_i}{\sum_i a^d_i} = \frac{\sum_j s_j a^s_j}{\sum_j a^s_j}
```

This property holds when both grids cover the same total area. The areas can be computed
from the regridder via row and column sums of ``A``:

```math
a^d_i = \sum_j A_{ij} \quad \text{and} \quad a^s_j = \sum_i A_{ij}
```

### Implementation Details

In ConservativeRegridding.jl:

- **Sparse storage**: The intersection matrix ``A`` is stored as a sparse matrix since most grid cell pairs do not intersect.
- **STRtree acceleration**: A Sort-Tile-Recursive tree enables efficient spatial queries to find intersecting cell pairs.
- **GeometryOps integration**: Polygon intersections and area calculations are handled by [GeometryOps.jl](https://github.com/JuliaGeo/GeometryOps.jl).
- **Antimeridian handling**: Polygons crossing the antimeridian are automatically fixed via `GeometryOps.fix`.
- **Normalization**: By default, the intersection areas are normalized to improve numerical conditioning.

## API Reference

```@docs
ConservativeRegridding.Regridder
ConservativeRegridding.regrid!
ConservativeRegridding.regrid
Base.transpose(::ConservativeRegridding.Regridder)
```

