# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Project Overview

ConservativeRegridding.jl performs area-weighted conservative regridding between polygon grids. "Conservative" means the area-weighted mean is preserved. The package computes intersection areas between source/destination grid cell pairs to create averaging weights.

## Common Commands

**Important**: Always use `julia --project=docs` when running scripts to access all dependencies.

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
julia --project=docs -e 'using ConservativeRegridding'
```

The project uses separate Project.toml files in `test/`, `docs/`, and `examples/`.

## Architecture

### Core Types and Functions

**`Regridder`** (`src/regridder/regridder.jl`): Main type storing intersection areas as a sparse matrix, grid cell areas, and work arrays.

- `transpose(regridder)` returns a regridder for the reverse direction (shares underlying data)
- `normalize!` scales the intersection matrix by its maximum value

**`Regridder(dst, src; normalize=true, intersection_operator=..., threaded=True())`**: Constructor that:
1. Determines manifold (Planar or Spherical) from input grids
2. Converts grids to spatial trees via `Trees.treeify`
3. Performs multithreaded dual depth-first search to find intersecting polygon pairs
4. Computes intersection areas via `DefaultIntersectionOperator` (user-overridable)

**`regrid!(dst_field, regridder, src_field)`** (`src/regridder/regrid.jl`): Performs regridding: `dst = (A * src) / dst_areas`

**`DefaultIntersectionOperator(manifold)`**: Dispatches to the appropriate intersection algorithm:
- Planar: `FosterHormannClipping`
- Spherical: `ConvexConvexSutherlandHodgman`

### Trees Submodule (`src/trees/`)

Provides quadtree-based spatial indexing for matrix-shaped grids.

**`treeify(manifold, grid)`** (`src/trees/interfaces.jl`): Converts grid representations into `SpatialTreeInterface`-compliant trees:
- Matrices of polygons -> `ExplicitPolygonGrid` + `TopDownQuadtreeCursor`
- Matrices of points -> `CellBasedGrid` + `TopDownQuadtreeCursor`
- Iterables of polygons -> `FlatNoTree`
- Existing spatial trees -> pass-through

**Grid types** (`src/trees/grids.jl`), all parameterized by manifold `M`:
- `ExplicitPolygonGrid{M}`: Wraps a matrix of pre-computed polygons
- `CellBasedGrid{M}`: Builds polygons on-the-fly from corner points
- `RegularGrid{M}`: For regular lon/lat grids defined by 1D vectors

**`AbstractCurvilinearGrid`** interface (`src/trees/interfaces.jl`):
- `getcell(grid, i, j)` -> polygon at grid position
- `ncells(grid, dim)` -> number of cells in dimension
- `cell_range_extent(grid, irange, jrange)` -> bounding extent for cell range

**Cursor types** (`src/trees/quadtree_cursors.jl`):
- `QuadtreeCursor`: Cursor with explicit index ranges
- `TopDownQuadtreeCursor`: Top-level cursor wrapping a grid
Sit on top of `AbstractCurvilinearGrid`s to provide quadtree descent.

**Wrappers** (`src/trees/wrappers.jl`):
- `KnownFullSphereExtentWrapper`: Avoids expensive extent computation for full-sphere trees

### Manifold Support

Grids operate on manifolds from GeometryOps:
- `Planar()`: Cartesian 2D, uses `Extents.Extent` for bounding boxes
- `Spherical()`: Unit sphere (lon/lat), uses `SphericalCap` for bounding regions

The manifold affects extent computation, intersection algorithms, and area calculations.

### Multithreading

`multithreaded_dual_depth_first_search` (`src/utils/MultithreadedDualDepthFirstSearch.jl`) provides parallel tree traversal, spawning tasks when nodes satisfy an area criterion.

### Package Extensions

- **ConservativeRegriddingOceananigansExt**: Oceananigans.jl grid integration
- **ConservativeRegriddingClimaCoreExt**: ClimaCore.jl grid integration (cubed sphere)
- **ConservativeRegriddingInterfacesExt**: Interfaces.jl contracts

### Key Dependencies

- **GeometryOps**: Polygon intersection, area calculations, `SpatialTreeInterface`
- **GeoInterface**: Geometry type wrappers
- **SparseArrays**: Sparse matrix storage for intersection weights
- **StableTasks/ChunkSplitters**: Multithreaded computation

### Mathematical Model

Given intersection matrix A, source field s, destination field d, and area vectors a_d, a_s:
- Forward: `d = (A * s) / a_d`

Grid cell areas are derived from row/column sums of A when grids fully overlap.
