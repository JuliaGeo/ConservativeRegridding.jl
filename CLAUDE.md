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

Key behaviors:
- `transpose(regridder)` returns a regridder for reverse direction (shares underlying data)
- `normalize!` scales intersection matrix by its maximum value

### Core Functions

**`Regridder(dst_vertices, src_vertices; normalize=true, ...)`**: Constructor that:
1. Wraps vertex arrays into `GeoInterface.Polygon` objects
2. Builds STRtrees for spatial indexing via SortTileRecursiveTree
3. Performs dual depth-first search to find intersecting polygon pairs
4. Computes intersection areas via `GeometryOps.intersection` by default, user-overridable by `intersection_operator` kwarg

**`regrid!(dst_field, regridder, src_field)`** (`src/regridder/regrid.jl`): Performs the regridding operation: `dst = (A * src) / dst_areas`

**`intersection_operator`**: The `Regridder` constructor accepts a custom `intersection_operator` function to compute the "intersection area" between two polygons, enabling non-standard intersection semantics.

### Trees Submodule (`src/trees/`)

The `Trees` submodule provides quadtree-based spatial indexing for matrix-shaped grids:

**`AbstractQuadtree`**: Abstract type for quadtree representations. Implement `getcell(quadtree, i, j)` and `ncells(quadtree, dim)`.

**Concrete implementations:**
- `ExplicitPolygonQuadtree`: Wraps a matrix of pre-computed polygons
- `CellBasedQuadtree`: Builds polygons on-the-fly from a matrix of corner points
- `RegularGridQuadtree`: For regular lon/lat grids defined by 1D x and y vectors

**`QuadtreeCursor`**: A cursor for traversing quadtrees, implementing GeometryOps' `SpatialTreeInterface` for dual depth-first search operations.

**`KnownFullSphereExtentWrapper`**: Wrapper for trees known to cover the entire sphere, avoiding expensive extent computation.

### Key Dependencies

- **GeometryOps**: Polygon intersection, area calculations, and `SpatialTreeInterface` for dual tree traversal
- **GeoInterface**: Geometry type wrappers
- **SortTileRecursiveTree**: STRtree spatial indexing for efficient intersection queries
- **SparseArrays**: Sparse matrix storage for intersection weights

### Mathematical Model

Given intersection matrix A, source field s, destination field d:
- Forward: `d = (A * s) / a_d`
- Backward: `s̃ = (Aᵀ * d) / a_s`

Grid cell areas are derived from row/column sums of A when grids fully overlap.
