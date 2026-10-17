# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ConservativeRegridding.jl performs area-weighted conservative regridding between polygon grids on planes or spheres. "Conservative" means the area-weighted mean is preserved during regridding. The package computes intersection areas between source/destination grid cell pairs, stores them as a sparse matrix, and uses that matrix for fast forward and backward regridding.

## Common Commands

**Important**: Always use `julia --project=docs` when running scripts and investigating — the docs environment has diagnostic packages not available in test.

```bash
# Run all tests
julia --project=test test/runtests.jl

# Run tests via Pkg
julia --project -e 'using Pkg; Pkg.test(; julia_args = ["--check-bounds=auto",])'

# Run a specific test file
julia --project=test -e 'include("test/usecases/simple.jl")'
julia --project=test -e 'include("test/regridding.jl")'
julia --project=test -e 'include("test/providers.jl")'

# Run examples/scripts or load the package interactively
julia --project=docs examples/speedy_to_speedy.jl
julia --project=docs -e 'using ConservativeRegridding'
```

The repo uses Julia's workspace feature (`[workspace]` in Project.toml) with separate environments in `test/`, `docs/`, and `examples/`.
`lib/ConservativeRegriddingTestHelpers/` is a workspace member subpackage of shared test fixtures/extensions (`TestHelpersClimaCoreExt`, `TestHelpersOceananigansExt`, `TestHelpersRingGridsExt`), used by `test/`.

## Architecture

### Three-Layer Design

```
Regridder (sparse matrix + area vectors)
    ↓ constructed by
Intersection Areas (dual DFS candidate search + parallel intersection computation)
    ↓ operates on
GridInterface grids (`ncells`, `cell_geometry`, `treeify`) — CR's own `CellBasedGrid`/`RegularGrid`/`ExplicitPolygonGrid`/`PolygonVectorGrid` and the extensions' providers
```

### Regridder (`src/regridder/`)

**`Regridder`** (`regridder.jl`): Stores a sparse intersection matrix, source/destination area vectors, and temporary work arrays.
- Constructor: `Regridder(dst, src)` auto-detects manifold, turns inputs into GridInterface grids (`asgrid`), treeifies them (`GridInterface.treeify(g; leaf_capacity = ForSweeps())`), runs dual DFS, computes intersections
- `transpose(regridder)` returns reverse-direction regridder sharing underlying data (no copy)
- `normalize!` scales intersection matrix by its maximum value

**`regrid!`** (`regrid.jl`): `dst = (A * src) / dst_areas`. Handles dense and non-contiguous arrays, copies to temporary arrays when needed for BLAS performance.

**`intersection_areas`** (`intersection_areas.jl`): Two-phase approach:
1. Dual DFS through spatial trees to find candidate cell pairs (extent-based pruning)
2. Parallel COO assembly on candidates, partitioned into `nthreads * 4` chunks via ChunkSplitters, fetched and `vcat`-reduced into a `SparseMatrixCSC`

**Intersection operator interface** (one trait + three hooks, all with defaults; the operator is the single extensible concept that drives assembly):
- `IntersectionReturnStyle(op)` — `OutOfPlaceSingleResult` (default; `op(src_cell, dst_cell) -> area`, driver stores the triplet) vs `InPlace` (`op(rows, cols, vals, item, src_tree, dst_tree)` pushes its own COO). Resolved once at the top of `intersection_areas` and threaded through.
- `work_items(op, candidate_pairs)` — units of parallel work (default: one candidate pair each).
- `output_matrix_size(op, src, dst)` — assembled `(nrows, ncols)` (default: `(ncells(dst), ncells(src))`).
- `output_eltype(op, [src, dst])` — element type of the assembled matrix (default: `Float64`); override to store something other than areas, e.g. a matrix of intersection polygons.
- The whole interface is `@public` (including `output_eltype` and `should_store_result`).

**Concrete operators** (`intersection_operators/`, one file each):
- `DefaultIntersectionOperator(manifold, src, dst)` dispatches on manifold and the grids' `cellshape` (Planar: `FosterHormannClipping`; Spherical: `ConvexConvexSutherlandHodgman` when both grids are `ConvexCells`, `FosterHormannClipping(Spherical())` otherwise) and uses all defaults, storing intersection areas.
- `IntersectionGridOperator` stores the raw intersection *polygons* instead of areas (overrides `output_eltype`/`should_store_result`); requires `normalize = false`. Demo in `examples/intersection_grid_assembler.jl`.

### Grids (`src/grids.jl`)

CR regrids [GridInterface](../GridInterface.jl) grids. `asgrid(x)` is the grid `Regridder` regrids for an input: identity on a GridInterface grid; CR's own grids for bare inputs — `CellBasedGrid(manifold, points)` for an (n+1)×(m+1) corner matrix, `ExplicitPolygonGrid` for a polygon matrix, `RegularGrid(manifold, x, y)` for an axis pair, `PolygonVectorGrid` (R-tree fallback tree) for a polygon vector; the extensions define it for their native objects. The engine reads `ncells`, `cell_geometry`, `cell_area`, `cellshape` and `treeify(grid; leaf_capacity = ForSweeps())` (`GridInterface.Structured`'s `TopDownQuadtreeCursor` for structured grids). A grid paired with a tree you already have is `GridInterface.GridAndTree(grid, tree)`.

### Manifold Support

Grids operate on manifolds from GeometryOps:
- `Planar()`: Cartesian 2D, uses `Extents.Extent` for bounding boxes
- `Spherical()`: Unit sphere (lon/lat), uses `SphericalCap` for bounding regions

The manifold affects extent computation, intersection algorithms, and area calculations. If source and destination have different manifolds, it promotes to Spherical.

### Multithreading

`multithreaded_dual_query` (`src/utils/MultithreadedDualDepthFirstSearch.jl`): Parallel dual-tree traversal. Splits the root node pair into `nthreads * chunks_per_thread` pairs by estimated work (`GridInterface.split_weight`), runs one task per pair, and concatenates the results in DFS order, so the output matches the serial traversal. Intersection area computation is separately parallelized via ChunkSplitters partitioning.

### Package Extensions (`ext/`)

All follow the same pattern: a GridInterface provider for the native grid type (a `Structured.AbstractStructuredGrid` with face hooks) plus `ConservativeRegridding.asgrid(native) = provider`, and the field plumbing (`extract_*_arraylike`, `initialize_regridding!`, `finalize_regridding!`).

- **OceananigansExt**: LatitudeLongitudeGrid, RectilinearGrid → `OceananigansLatLonGrid`; TripolarGrid, RotatedLatitudeLongitudeGrid → `OceananigansCurvilinearGrid` (a U-pivot grid regrids as `FoldedLayoutGrid`: the field layout with the fold row's duplicates as zero-area ghosts); ConformalCubedSphereGrid → `OceananigansCubedSphereGrid`
- **ClimaCoreExt**: Cubed sphere topologies → `ClimaCoreCubedSphereGrid` (6 faces, element order of the topology). Also provides custom `Regridder` constructors for spectral-element (SE) spaces: SE↔FV regridding is assembled by two `InPlace` intersection operators (`SEToFVIntersectionOperator`, `FVToSEIntersectionOperator`) that plug into the shared `intersection_areas` driver — FV→SE overrides `work_items` to make each element (not each candidate pair) the unit of parallel work for its local mass-matrix solve.
- **HealpixExt**, **RingGridsExt**: HEALPix and SpeedyWeather grids
- **SpeedyWeatherExt**: Requires *both* RingGrids and SpeedyWeather loaded (dual weak dependency)
- **InterfacesExt**: Interfaces.jl contracts

### API Surface

The package uses `@public` from SciMLPublic for API visibility: `Regridder`, `regrid`, `regrid!`, `areas` are public; so is the N-D regridding interface (`AbstractDimensionalSlicer`, `NDSliceLoop`, `slice_views`, `extract_source_arraylike`, `extract_dest_arraylike`); so is the intersection-operator interface (`intersection_areas`, `DefaultIntersectionOperator`, `IntersectionGridOperator`, `IntersectionReturnStyle`, `OutOfPlaceSingleResult`, `InPlace`, `work_items`, `output_matrix_size`, `output_eltype`, `should_store_result`). `asgrid`, `PolygonVectorGrid` and the re-exported `CellBasedGrid`, `RegularGrid`, `ExplicitPolygonGrid` are exported.
