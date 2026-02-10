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
julia --project=test -e 'include("test/trees/grids.jl")'
julia --project=test -e 'include("test/trees/quadtree_cursors.jl")'

# Run examples/scripts or load the package interactively
julia --project=docs examples/speedy_to_speedy.jl
julia --project=docs -e 'using ConservativeRegridding'
```

The repo uses Julia's workspace feature (`[workspace]` in Project.toml) with separate environments in `test/`, `docs/`, and `examples/`.

**Test dependency note**: `test/Project.toml` pins Oceananigans to a specific fork (`briochemc/Oceananigans.jl` branch `FPivot`). If test instantiation fails on Oceananigans, check this source entry.

## Architecture

### Three-Layer Design

```
Regridder (sparse matrix + area vectors)
    ↓ constructed by
Intersection Areas (dual DFS candidate search + parallel intersection computation)
    ↓ operates on
Trees Module (grid representations + quadtree cursors for spatial indexing)
```

### Regridder (`src/regridder/`)

**`Regridder`** (`regridder.jl`): Stores a sparse intersection matrix, source/destination area vectors, and temporary work arrays.
- Constructor: `Regridder(dst, src)` auto-detects manifold, treeifies grids, runs dual DFS, computes intersections
- `transpose(regridder)` returns reverse-direction regridder sharing underlying data (no copy)
- `normalize!` scales intersection matrix by its maximum value

**`regrid!`** (`regrid.jl`): `dst = (A * src) / dst_areas`. Handles dense and non-contiguous arrays, copies to temporary arrays when needed for BLAS performance.

**`intersection_areas`** (`intersection_areas.jl`): Two-phase approach:
1. Dual DFS through spatial trees to find candidate cell pairs (extent-based pruning)
2. Parallel intersection area computation on candidates, partitioned into `nthreads * 4` chunks via ChunkSplitters

**Intersection operators** dispatch on manifold:
- Planar: `FosterHormannClipping`
- Spherical: `ConvexConvexSutherlandHodgman`

### Trees Module (`src/trees/Trees.jl`)

**Grid types** (`grids.jl`), all `<: AbstractCurvilinearGrid`, parameterized by manifold `M`:
- `ExplicitPolygonGrid{M}`: Wraps a matrix of pre-computed polygons
- `CellBasedGrid{M}`: Builds polygons on-the-fly from (n+1)×(m+1) corner point matrix
- `RegularGrid{M}`: Regular lon/lat grids from 1D coordinate vectors

**Required interface** for `AbstractCurvilinearGrid`: `getcell(grid, i, j)`, `ncells(grid, dim)`, `cell_range_extent(grid, irange, jrange)`

**Quadtree cursors** (`quadtree_cursors.jl`): Implement `SpatialTreeInterface` on top of grids:
- `TopDownQuadtreeCursor`: Recursive subdivision by index ranges (primary cursor used)
- `QuadtreeCursor`: Bottom-up traversal with level-based indexing

**Specialized cursors** (`specialized_quadtree_cursors.jl`):
- `IndexOffsetQuadtreeCursor`: For multi-grid scenarios (e.g., cubed spheres), applies an index offset to grid-local indices
- `ReorderedTopDownQuadtreeCursor`: For custom element orderings (space-filling curves in ClimaCore)

**Wrappers** (`wrappers.jl`):
- `KnownFullSphereExtentWrapper`: Returns full-sphere extent to skip expensive extent computation
- `CubedSphereToplevelTree`: Vector of per-face cursors with global indexing

**`treeify(manifold, grid)`** (`interfaces.jl`): Dispatches input to the right tree type:
- Matrices of polygons → `TopDownQuadtreeCursor(ExplicitPolygonGrid(...))`
- Matrices of points → `TopDownQuadtreeCursor(CellBasedGrid(...))`
- Iterables of polygons → `FlatNoTree`
- Tuple of vectors → `TopDownQuadtreeCursor(RegularGrid(...))`
- Existing spatial trees → pass-through

### Manifold Support

Grids operate on manifolds from GeometryOps:
- `Planar()`: Cartesian 2D, uses `Extents.Extent` for bounding boxes
- `Spherical()`: Unit sphere (lon/lat), uses `SphericalCap` for bounding regions

The manifold affects extent computation, intersection algorithms, and area calculations. If source and destination have different manifolds, it promotes to Spherical.

### Multithreading

`multithreaded_dual_query` (`src/utils/MultithreadedDualDepthFirstSearch.jl`): Parallel dual-tree traversal. Spawns tasks when both nodes are leaves or when nodes satisfy an area criterion (avoids spawning excessive tasks for small regions). Intersection area computation is separately parallelized via ChunkSplitters partitioning.

### Package Extensions (`ext/`)

All follow the same pattern: implement `Trees.treeify()` for domain-specific grid types.

- **OceananigansExt**: LatitudeLongitudeGrid, RectilinearGrid, TripolarGrid → vertex matrices wrapped in KnownFullSphereExtentWrapper
- **ClimaCoreExt**: Cubed sphere topologies → per-face cursors with index remapping
- **HealpixExt**, **RingGridsExt**: HEALPix and SpeedyWeather grids
- **SpeedyWeatherExt**: Requires *both* RingGrids and SpeedyWeather loaded (dual weak dependency)
- **InterfacesExt**: Interfaces.jl contracts

### API Surface

The package uses `@public` from SciMLPublic for API visibility: `Regridder`, `regrid`, `regrid!`, `areas` are public. Grid types and tree types are exported.
