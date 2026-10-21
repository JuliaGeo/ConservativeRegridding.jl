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

```@example speedy
using SpeedyWeather
using ConservativeRegridding

# Create random data on two different geodesic grid types
field1 = rand(OctaHEALPixGrid, 4)
field2 = rand(OctaminimalGaussianGrid, 4)
```

```@example speedy
# Build the regridder (precomputes intersection areas). The RingGrids extension
# answers for the fields' grids, so they are passed as they are.
R = ConservativeRegridding.Regridder(field1, field2)
```

```@example speedy
# Regrid from field2 to field1
ConservativeRegridding.regrid!(field1, R, field2)
```

```@example speedy
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

A dual-tree search over the two grids' spatial trees finds the cell pairs that may
intersect, so only those are clipped; see [How it works](how_it_works.md).

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
- **Grids and trees**: Both inputs are [GridInterface](interfaces/grids.md) grids, each with a spatial tree from `GridInterface.treeify`; a dual-tree search of the two trees names the candidate cell pairs.
- **GeometryOps integration**: Polygon intersections and area calculations are handled by [GeometryOps.jl](https://github.com/JuliaGeo/GeometryOps.jl).
- **Normalization**: `Regridder(dst, src; normalize = true)` rescales ``A`` and both area vectors by the largest entry of ``A``, for numerical conditioning; the weights of `regrid!` are unchanged.

## API Reference

```@docs
ConservativeRegridding.Regridder
ConservativeRegridding.regrid!
ConservativeRegridding.areas
Base.transpose(::ConservativeRegridding.Regridder)
ConservativeRegridding.AbstractRegridder
ConservativeRegridding.save_esmf_weights
```

### Grids

What `Regridder` makes of its inputs; see [Grid Interface](interfaces/grids.md) for
regridding your own grid type.

```@docs
ConservativeRegridding.asgrid
ConservativeRegridding.PolygonVectorGrid
```

### Candidate search

```@docs
ConservativeRegridding.candidate_pairs
ConservativeRegridding.WeightedDualDepthFirstSearch
ConservativeRegridding.weighted_dual_depth_first_search
ConservativeRegridding.MultithreadedDualDepthFirstSearch.multithreaded_dual_query!
ConservativeRegridding.MultithreadedDualDepthFirstSearch.frontier
ConservativeRegridding.MultithreadedDualDepthFirstSearch.MAX_EXTRA_PAIRS
```

### Intersection operator interface

These functions let you plug a custom weight kernel into the shared parallel assembly.
See [Customizing the weights: the intersection operator interface](@ref) for a walkthrough.

```@docs
ConservativeRegridding.intersection_areas
ConservativeRegridding.IntersectionReturnStyle
ConservativeRegridding.OutOfPlaceSingleResult
ConservativeRegridding.InPlace
ConservativeRegridding.work_items
ConservativeRegridding.output_matrix_size
ConservativeRegridding.output_eltype
ConservativeRegridding.should_store_result
ConservativeRegridding.task_local_operator
ConservativeRegridding.SparseMatrixAssemblyCache
```

These are the various intersection operators already implemented:

```@docs
ConservativeRegridding.DefaultIntersectionOperator
ConservativeRegridding.ClipKind
ConservativeRegridding.ConvexClip
ConservativeRegridding.GeneralClip
ConservativeRegridding.clip_kind
ConservativeRegridding.DefaultIntersectionFailureError
ConservativeRegridding.IntersectionGridOperator
ConservativeRegridding.intersection_polygon_type
```

and there are two more in the ClimaCore extension, for spectral element based regridding.

### Example fields

Analytic test fields, evaluated at a cell's longitude and latitude, for checking a
regridder against a known function.

```@docs
ConservativeRegridding.ExampleFieldFunction
ConservativeRegridding.LongitudeField
ConservativeRegridding.SinusoidField
ConservativeRegridding.HarmonicField
ConservativeRegridding.GulfStreamField
ConservativeRegridding.VortexField
```