# ConservativeRegridding.jl changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Interface hooks generalizing sparse-matrix assembly for custom intersection operators:
  `IntersectionReturnStyle` (`OutOfPlaceSingleResult`/`InPlace`), `work_items`,
  `output_matrix_size`, and `output_eltype`, so extensions like ClimaCore only need to
  define their own intersection operator rather than the full assembly loop (#110).
- `output_eltype` lets an intersection operator customize the element type stored in the
  assembled matrix; an example is included that constructs an "intersection grid" — a
  sparse matrix of the intersection polygons themselves, rather than their areas (#121).
- Native HEALPix and OctaHEALPix grid support in the RingGrids (SpeedyWeather) extension (#122).

### Changed
- ClimaCore SE↔FV sparse-matrix assembly reimplemented on the new `InPlace`
  intersection-operator interface and parallelized (#110).

### Fixed
- Type instability in the ClimaCore extension's `spherical_triangle_area`.

## [0.2.5] - 2026-06-11

### Changed
- Spherical `cell_range_extent` now builds bounding caps from a lazy `PerimeterPoints`
  iterator instead of a materialized vector, removing a per-call allocation in the
  dual-DFS hot path (#106).
- Replaced per-grid-type `should_parallelize` overrides with a single leaf-count-based
  default for quadtree cursors (spawns a parallel task once a subtree's leaf count drops
  below `total_cells / (nthreads * 32)`); `WithParallelizePolicy` remains available as an
  instance-level override (#106).
- Widened Oceananigans compat to include 0.110 (#113).

## [0.2.4] - 2026-06-04

### Added
- Conservative regridding between ClimaCore spectral-element (SE) cubed-sphere spaces
  and finite-volume (FV) grids, in both directions, via the standard `Regridder`
  constructor (#99).
- SE↔FV assembly integrates the true polygon intersection between SE elements and FV
  cells using triangulated Gauss/Dunavant quadrature (new `TriangleQuadrature` module),
  rather than simple node-in-cell binning.
- FV→SE regridding solves a per-element L2 mass-matrix projection for higher-order
  accuracy and exact conservation of constants; direct SE→SE regridding isn't supported
  yet (chain SE→FV→SE through two regridders).
- ClimaCore extension helpers for converting between fields and flat nodal vectors:
  `se_field_to_vec`, `vec_to_se_field!`, `se_node_positions`, `se_node_weights`.

### Changed
- ClimaCore extension's SE basis evaluation now uses ClimaCore's own `Quadratures`
  interpolation-matrix utilities instead of a custom Lagrange-basis implementation.

## [0.2.3] - 2026-05-28

### Changed
- Widened Oceananigans compat to include 0.109 (#104).

## [0.2.2] - 2026-05-26

### Added
- ESMF offline-weights export via a new NCDatasets-gated extension:
  `save_esmf_weights(path, regridder; ...)` writes the standard ESMF/xESMF NetCDF weight
  format (#88).
- Regridding support for SpeedyWeather's regular "Full" grids (e.g. `FullClenshawGrid`,
  `FullGaussianGrid`) via `RingGrids.AbstractFullGrid` (#90).
- Public `Trees.should_parallelize(tree, node, extent)` API and `WithParallelizePolicy`
  wrapper to customize parallel task-spawning granularity per tree type, replacing the
  internal `_area_criterion` (planar `Extents.Extent` trees still need a user-supplied
  method; a manifold-agnostic default arrived later in #106) (#94).
- `regrid!` split into a public 5-step pipeline (`extract_source_arraylike`,
  `extract_dest_arraylike`, `initialize_regridding!`, `perform_regridding!`,
  `finalize_regridding!`) that extensions hook into individually (#102).
- Public `AbstractDimensionalSlicer`/`slice_views` interface and built-in `NDSliceLoop`
  slicer, re-enabling `regrid!(dst, regridder, src; dims = k)` for N-dimensional
  `StridedArray` inputs, with clear errors on invalid `dims` or mismatched non-spatial
  axes (#102).
- `regrid!` now works directly on `ClimaCore.Fields.Field` via the new pipeline
  (integrate-over-element on read, per-element broadcast on write) (#102).

### Changed
- `Regridder` construction no longer normalizes the intersection matrix by its maximum
  value by default (`normalize = false`), since ClimaCore's spectral-element pipeline
  needs unscaled areas (#102).
- Healpix extension's `regrid!` glue ported onto the new 5-function pipeline (#102).
- Oceananigans extension: replaced the post-construction sparse-matrix "area-halving"
  trick for folded Tripolar grids with simpler value-mirroring in
  `finalize_regridding!` (#102).
- Widened Oceananigans compat to 0.107, 0.108 (#103).

### Fixed
- `NaN`/incorrect values at duplicated fold-row cells when regridding onto an
  Oceananigans `RightCenterFolded` Tripolar grid, by mirroring each primary cell's
  regridded value onto its duplicate partner slot via a GPU-compatible kernel (#101).

## [0.2.1] - 2026-04-13

### Added
- Cross-grid "sweat test" that regrids every supported grid type against every other to
  catch cross-cutting bugs (#46).
- XESMF.jl (Python xESMF/ESMF) comparison test validating weights and regridded fields
  to ~1e-14 relative error (#77).

### Changed
- Optimized `_compute_cell_matrix!` (Oceananigans extension) loop order for
  column-major cache locality (#71).
- Unified Tripolar grid handling now that upstream Oceananigans restored
  `RightFaceFolded`'s full prognostic domain, removing the special-cased
  `FPivotTripolarGrid` path (#79).
- Bumped Oceananigans compat progressively to 0.107 (#73, #79, #85).

### Fixed
- `regrid!` rejecting `Matrix`/higher-dimensional array input despite the docstring
  promising N-dimensional support, via a new `dims` keyword (#68).
- Swapped `mapreduce` arguments in `cell_range_extent` for `ExplicitPolygonGrid{Planar}`
  that made constructing Planar regridders throw a `MethodError` (#70).

## [0.2.0] - 2026-02-25

### Added
- New `Trees` module for spatial indexing: `AbstractCurvilinearGrid` grid types
  (`ExplicitPolygonGrid`, `CellBasedGrid`, `RegularGrid`) and quadtree cursors
  (`TopDownQuadtreeCursor`, `QuadtreeCursor`) behind a generic `SpatialTreeInterface`,
  replacing the flat `SortTileRecursiveTree`-only approach (#34).
- First-class Spherical manifold support (spherical-cap tree extents, convex
  spherical-polygon clipping) alongside Planar; `Regridder` now auto-detects and
  promotes the manifold from its input grids (#34).
- Multithreaded regridder construction: parallel dual depth-first search plus chunked
  intersection-area computation via StableTasks/ChunkSplitters, toggleable with a
  `threaded` keyword (#34).
- Customizable `intersection_operator` argument to override the default per-manifold
  area-computation algorithm (#34).
- Package extensions: ClimaCore (cubed-sphere spectral-element topologies, including
  space-filling-curve element orderings), Oceananigans (`LatitudeLongitudeGrid`,
  `RectilinearGrid`, `TripolarGrid`, `ImmersedBoundaryGrid`), Healpix.jl (native HEALPix
  pixelizations), Interfaces.jl, plus initial RingGrids/SpeedyWeather scaffolding
  (#34, #43).
- `on_architecture` support for `Regridder`, moving its matrix/areas/work buffers across
  Oceananigans architectures (e.g. to/from GPU) (#47).
- Declared a public API surface via SciMLPublic's `@public`
  (`Regridder`, `regrid`, `regrid!`, `areas`) (#34).

### Changed
- `regrid!` now routes non-contiguous source/destination arrays through dense temporary
  buffers owned by the `Regridder` to keep the sparse matrix-vector product fast.
- Source layout reorganized into `src/regridder/` and `src/trees/`; tests split into
  `test/usecases/`, `test/extensions/`, and `test/trees/` with combinatorial
  per-grid-type/size coverage (#34).

### Fixed
- Tripolar-grid fold-row bugs (`RightFaceFolded`/`RightCenterFolded` topologies),
  including a padded/ghost-cell tree wrapper so fold rows no longer double-count
  intersections (#60).
- Longitude-range mismatch when regridding between ClimaCore cubed-sphere and
  Oceananigans tripolar grids (#59).
- GPU scalar-indexing errors when regridding ClimaCore fields defined on GPU (#56).
- `best_manifold` detection for `ImmersedBoundaryGrid`, now delegating to its underlying
  grid (#57).

## [0.1.0] - 2025-12-12

### Added
- `Regridder` type storing a sparse intersection-area matrix plus source/destination
  cell-area vectors, built via dual depth-first search over
  `SortTileRecursiveTree.STRtree`s with polygon intersection/area from GeometryOps.
- `regrid!`/`regrid` for conservative, mean-preserving regridding
  (`dst = (A * src) / dst_areas`), broadcasting across extra dimensions for
  N-dimensional fields (#11).
- `transpose(regridder)` for backward regridding, sharing the same underlying arrays
  with no copy (#7).
- `normalize!`, applied by default, scaling the intersection matrix and area vectors by
  their maximum value (#11).
- Antimeridian-crossing polygons passed through `GeometryOps.fix` before intersection
  (best-effort; some meridian bugs remained and were fixed in later releases) (#17, #19).
- Test coverage against ClimaCore, Oceananigans, and SpeedyWeather grids/fields as
  test-only dependencies (no package extensions yet) (#4, #29).
- Initial docs site and README covering usage and the regridding math (#13, #30).
- Relicensed to MIT and registered in the Julia General registry (#12).

[Unreleased]: https://github.com/JuliaGeo/ConservativeRegridding.jl/compare/v0.2.5...HEAD
[0.2.5]: https://github.com/JuliaGeo/ConservativeRegridding.jl/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/JuliaGeo/ConservativeRegridding.jl/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/JuliaGeo/ConservativeRegridding.jl/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/JuliaGeo/ConservativeRegridding.jl/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/JuliaGeo/ConservativeRegridding.jl/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/JuliaGeo/ConservativeRegridding.jl/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/JuliaGeo/ConservativeRegridding.jl/releases/tag/v0.1.0
