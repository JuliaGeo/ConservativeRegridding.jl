module ConservativeRegridding

using DocStringExtensions
import LinearAlgebra
import GeoInterface
import GeometryOps
import SortTileRecursiveTree
import Extents
import SparseArrays
import ChunkSplitters
import StableTasks
import ProgressMeter

using GeometryOpsCore: booltype, BoolsAsTypes, True, False, istrue
using GeometryOpsCore: Manifold, Planar, Spherical

using SciMLPublic: @public

include("utils/example_data.jl")
export ExampleFieldFunction, LongitudeField, SinusoidField, HarmonicField, GulfStreamField, VortexField

include("trees/Trees.jl")
using .Trees

# The serial traversal and its task frontier both dispatch on `Trees.split_weight`.
include("utils/WeightedDualDepthFirstSearch.jl")
using .WeightedDualDepthFirstSearch

include("utils/MultithreadedDualDepthFirstSearch.jl")
using .MultithreadedDualDepthFirstSearch

export AbstractCurvilinearGrid, ncells, cell_index_count, getcell
export ExplicitPolygonGrid, CellBasedGrid, RegularGrid
export QuadtreeCursor, TopDownQuadtreeCursor
export should_parallelize, WithParallelizePolicy
export split_weight

include("regridder/triangle_quadrature.jl")

include("regridder/regridder.jl")
include("regridder/regrid.jl")
include("regridder/sparse_assembly.jl")
include("regridder/intersection_areas.jl")
# Intersection operators, implementing the interface defined in intersection_areas.jl
include("regridder/intersection_operators/default_intersection_operator.jl")
include("regridder/intersection_operators/intersection_grid_operator.jl")


@public Regridder, regrid, regrid!
@public areas
@public AbstractDimensionalSlicer, NDSliceLoop, slice_views, extract_source_arraylike, extract_dest_arraylike
# Intersection-operator assembly interface (operators can plug into `intersection_areas`)
@public intersection_areas, SparseMatrixAssemblyCache, DefaultIntersectionOperator, IntersectionGridOperator
@public IntersectionReturnStyle, OutOfPlaceSingleResult, InPlace
# Deterministic weight-guided traversal.
@public weighted_dual_depth_first_search
@public work_items, output_matrix_size, output_eltype, should_store_result, task_local_operator

"""
    save_esmf_weights(path, regridder;
        src_grid_name="source", dst_grid_name="destination",
        src_shape=nothing, dst_shape=nothing,
        created_at=nothing) -> path

Write `regridder`'s sparse weights to an ESMF offline-weights NetCDF file at `path`.
Requires `NCDatasets.jl` to be loaded (activates the extension).

## Format (ESMF convention)

| Variable  | Dim      | Description |
|-----------|----------|-------------|
| `S`       | `(n_s,)` | Weight: `intersection_area / dst_cell_area` |
| `row`     | `(n_s,)` | Destination cell index (1-based) |
| `col`     | `(n_s,)` | Source cell index (1-based) |
| `frac_a`  | `(n_a,)` | Fraction of source cell area covered by destination grid |
| `frac_b`  | `(n_b,)` | Fraction of destination cell area covered by source grid |
| `area_a`  | `(n_a,)` | Source cell areas |
| `area_b`  | `(n_b,)` | Destination cell areas |

Uses **destarea** normalization (the ESMF/xESMF default). The `Regridder`
must be built with `normalize=false`; `normalize=true` rescales the
intersection matrix and the exported weights will not match ESMF conventions.
For full-sphere-to-full-sphere pairs, `frac_a` and `frac_b` should be 1.0
to machine precision.

`src_shape`/`dst_shape` (e.g. `(720, 361)`, `(90, 90, 6)`) and the grid
name strings are stored as global attributes for provenance. Pass
`created_at` (e.g. `string(Dates.now())`) to stamp a creation timestamp;
omit for reproducible (byte-identical) output.
"""
function save_esmf_weights end

end
