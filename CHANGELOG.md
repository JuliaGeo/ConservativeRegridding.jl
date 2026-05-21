# ConservativeRegridding.jl changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `AbstractDimensionalSlicer` abstract type and `slice_views` interface, enabling
  extensions to plug N-D field layouts into the regridding pipeline without
  defining new `regrid!` dispatches. The extractor hooks
  `extract_source_arraylike` and `extract_dest_arraylike` are now public for
  extension authors.
- `NDSliceLoop` built-in slicer for N-dimensional `StridedArray` inputs.
- `regrid!(dst, regridder, src; dims = k)` now works for N-D `StridedArray`
  inputs (re-introducing N-D support via the new interface; package field types
  like `Oceananigans.Field` are unaffected because they aren't `StridedArray`).
  Invalid `dims` values and mismatched non-spatial axes now throw clear errors.
