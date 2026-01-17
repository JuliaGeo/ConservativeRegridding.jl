# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ConservativeRegridding.jl is a Julia package for area-weighted conservative regridding between two polygon grids. "Conservative" means the area-weighted mean is preserved during regridding. The package computes intersection areas between all source/destination grid cell pairs to create weights for averaging.

## Common Commands

```bash
# Run all tests
julia --project=test test/runtests.jl

# Run tests via Pkg
julia --project -e 'using Pkg; Pkg.test(; julia_args = ["--check-bounds=auto",])'

# Run a specific test file
julia --project=test -e 'include("test/usecases/simple.jl")'
julia --project=test -e 'include("test/trees/grids.jl")'
julia --project=test -e 'include("test/trees/quadtree_cursors.jl")'

# Start Julia REPL with the package loaded
julia --project -e 'using ConservativeRegridding'
```

The project uses a workspace layout with separate Project.toml files in `test/`, `docs/`, and `examples/`.

## Architecture

### Core Types

**`Regridder`** (`src/regridder/regridder.jl`): The main type storing:
- `intersections`: Sparse matrix of intersection areas between source and destination grid cells
- `dst_areas` / `src_areas`: Vectors of grid cell areas
- `dst_temp` / `src_temp`: Work arrays for non-contiguous memory regridding

Key behaviors:
- `transpose(regridder)` returns a regridder for reverse direction (shares underlying data)
- `normalize!` scales intersection matrix by its maximum value

### Core Functions

**`Regridder(dst, src; normalize=true, intersection_operator=..., threaded=True())`**: Constructor that:
1. Determines manifold (Planar or Spherical) from input grids
2. Converts grids to spatial trees via `Trees.treeify`
3. Performs dual depth-first search to find intersecting polygon pairs (multithreaded by default)
4. Computes intersection areas via `DefaultIntersectionOperator`, user-overridable by `intersection_operator` kwarg

**`regrid!(dst_field, regridder, src_field)`** (`src/regridder/regrid.jl`): Performs the regridding operation: `dst = (A * src) / dst_areas`

**`DefaultIntersectionOperator(manifold)`**: Default intersection operator that dispatches to appropriate algorithm based on manifold:
- Planar: Uses `FosterHormannClipping`
- Spherical: Uses `ConvexConvexSutherlandHodgman`

### Trees Submodule (`src/trees/`)

The `Trees` submodule provides quadtree-based spatial indexing for matrix-shaped grids.

**`treeify(manifold, grid)`** (`src/trees/interfaces.jl`): Main entry point that converts any grid representation into a `SpatialTreeInterface`-compliant tree. Handles:
- Matrices of polygons → `ExplicitPolygonGrid` + `TopDownQuadtreeCursor`
- Matrices of points → `CellBasedGrid` + `TopDownQuadtreeCursor`
- Iterables of polygons → `FlatNoTree`
- Existing spatial trees (pass-through)

**`AbstractCurvilinearGrid`** (`src/trees/interfaces.jl`): Abstract type for grid representations. Implement:
- `getcell(grid, i, j)` → returns polygon at grid position
- `ncells(grid, dim)` → number of cells in dimension
- `cell_range_extent(grid, irange, jrange)` → bounding extent for cell range

**Concrete grid implementations** (`src/trees/grids.jl`):
- `ExplicitPolygonGrid{M}`: Wraps a matrix of pre-computed polygons
- `CellBasedGrid{M}`: Builds polygons on-the-fly from a matrix of corner points
- `RegularGrid{M}`: For regular lon/lat grids defined by 1D x and y vectors

All grid types are parameterized by manifold `M` (Planar or Spherical).

**`AbstractQuadtreeCursor`** (`src/trees/interfaces.jl`): Abstract type for quadtree traversal. Implements GeometryOps' `SpatialTreeInterface`.

**Cursor implementations** (`src/trees/quadtree_cursors.jl`):
- `QuadtreeCursor`: Cursor with explicit index ranges
- `TopDownQuadtreeCursor`: Top-level cursor wrapping a grid

**Tree wrappers** (`src/trees/wrappers.jl`):
- `KnownFullSphereExtentWrapper`: For trees known to cover the entire sphere, avoiding expensive extent computation

### Manifold Support

Grids can operate on different manifolds (from GeometryOps):
- `Planar()`: Cartesian 2D coordinates, uses `Extents.Extent` for bounding boxes
- `Spherical()`: Unit sphere coordinates (lon/lat), uses `SphericalCap` for bounding regions

The manifold affects:
- `cell_range_extent` computation (rectangular extents vs spherical caps)
- Intersection algorithm selection
- Area calculations

### Multithreading (`src/utils/MultithreadedDualDepthFirstSearch.jl`)

Provides `multithreaded_dual_depth_first_search` for parallel tree traversal. Spawns tasks when nodes satisfy an area criterion, enabling efficient parallel intersection computation.

### Package Extensions

- **ConservativeRegriddingOceananigansExt**: Integration with Oceananigans.jl grids
- **ConservativeRegriddingInterfacesExt**: Interfaces.jl contract definitions

### Key Dependencies

- **GeometryOps**: Polygon intersection, area calculations, and `SpatialTreeInterface` for dual tree traversal
- **GeoInterface**: Geometry type wrappers
- **SortTileRecursiveTree**: STRtree spatial indexing for efficient intersection queries
- **SparseArrays**: Sparse matrix storage for intersection weights
- **StableTasks/ChunkSplitters**: Multithreaded computation

### Mathematical Model

Given intersection matrix A, source field s, destination field d:
- Forward: `d = (A * s) / a_d`
- Backward: `s̃ = (Aᵀ * d) / a_s`

Grid cell areas are derived from row/column sums of A when grids fully overlap.
