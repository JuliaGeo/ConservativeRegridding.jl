# 2nd Order Conservative Regridding Design

## Overview

Add support for 2nd order conservative regridding to ConservativeRegridding.jl. The 2nd order method improves accuracy by incorporating gradient information from neighboring source cells, achieving O(2) accuracy for linear/quadratic fields compared to O(1) for 1st order.

## Algorithm

### Mathematical Foundation

For 1st order, each destination cell receives a simple area-weighted average:

```
dst = (A * src) / dst_areas
```

For 2nd order, the source field is approximated with a Taylor expansion:

```
f(r) = f̄_src + ∇f · (r - r_src)
```

Where:
- `f̄_src` = cell-averaged value in the source cell
- `∇f` = gradient computed from neighboring cell values via Green's theorem
- `r_src` = source cell centroid
- `r` = position within the overlap region

### Weight Calculation

For each overlap between source cell and destination cell:

**Source cell weight:**
```
w_src = (area_overlap / area_dst) - dot(diff_cntr, src_grad) * (area_overlap / area_dst)
```

**Neighbor weights (for each neighbor of the source cell):**
```
w_nbr = dot(diff_cntr, nbr_grad) * (area_overlap / area_dst)
```

Where:
- `diff_cntr` = overlap_centroid - source_centroid
- `src_grad` = gradient coefficient for source cell (from Green's theorem)
- `nbr_grad` = gradient coefficient for each neighbor (from Green's theorem)

### Gradient Computation via Green's Theorem

For each source cell:
1. Identify neighbors (cells sharing edges or vertices)
2. Sort neighbors counter-clockwise around source centroid
3. Form "neighbor polygon" from neighbor centroids
4. Apply Green's theorem to compute gradient coefficients

Validation checks before using gradients:
- At least 3 neighbors required
- Source centroid must be inside neighbor polygon
- Neighbor polygon must have non-zero area

If validation fails, silently fall back to 1st order (zero gradient) for that cell.

> **Future extension:** A `strict=true` keyword argument could be added to error instead of falling back silently.

## API Design

### Method Types

```julia
abstract type AbstractRegridMethod end

struct Conservative1stOrder <: AbstractRegridMethod end
struct Conservative2ndOrder <: AbstractRegridMethod end
struct NearestNeighbor <: AbstractRegridMethod end  # future
```

### Constructor

```julia
# User-facing API
Regridder(dst, src; method = Conservative1stOrder(), kwargs...)
Regridder(dst, src; method = Conservative2ndOrder(), kwargs...)

# Internal dispatch
Regridder(manifold, method::Conservative1stOrder, dst, src; kwargs...)
Regridder(manifold, method::Conservative2ndOrder, dst, src; kwargs...)
```

Default is `Conservative1stOrder()` for backwards compatibility.

### Regridder Struct

```julia
struct Regridder{M <: AbstractRegridMethod, W, A, V} <: AbstractRegridder
    method::M
    intersections::W
    dst_areas::A
    src_areas::A
    dst_temp::V
    src_temp::V
end
```

The method is stored as a type parameter for dispatch and as a field for introspection.

### Transpose Behavior

2nd order regridding is asymmetric. The weights encode source grid structure (neighbor gradients, centroids relative to source cells). Transposing the weight matrix does not produce valid 2nd order regridding in the reverse direction.

```julia
function LinearAlgebra.transpose(regridder::Regridder{<:Conservative2ndOrder})
    error("Cannot transpose a 2nd order regridder. " *
          "Build a separate Regridder(src, dst; method=Conservative2ndOrder()) for reverse direction.")
end

# 1st order can still transpose
function LinearAlgebra.transpose(regridder::Regridder{M}) where M <: Union{Conservative1stOrder, NearestNeighbor}
    Regridder(regridder.method, transpose(regridder.intersections),
              regridder.src_areas, regridder.dst_areas,
              regridder.src_temp, regridder.dst_temp)
end
```

## Storage Format

The 2nd order method uses the same sparse matrix format as 1st order, but with additional entries. A destination cell can receive contributions from source cells that don't directly overlap it (the neighbors used for gradient computation).

The `regrid!` function remains unchanged:
```julia
dst = (W * src) / dst_areas
```

## Adjacency Computation

Neighbors are cells sharing edges or vertices (8-connectivity for quad grids).

### Multiple Dispatch for Performance

```julia
# Generic: spatial queries for unstructured grids
function compute_adjacency(manifold, tree)
    # Dual DFS on tree against itself
    # Check if polygons touch (boundary intersection)
    # O(n log n) complexity
end

# Specialized: index arithmetic for structured grids
function compute_adjacency(manifold, tree::QuadtreeCursor{<:CellBasedQuadtree})
    ni, nj = Trees.ncells(tree)
    adjacency = Vector{Vector{Int}}(undef, ni * nj)

    for j in 1:nj, i in 1:ni
        idx = linear_index(i, j, ni)
        neighbors = Int[]
        for dj in -1:1, di in -1:1
            (di == 0 && dj == 0) && continue
            ni_new, nj_new = i + di, j + dj
            if 1 <= ni_new <= ni && 1 <= nj_new <= nj
                push!(neighbors, linear_index(ni_new, nj_new, ni))
            end
        end
        adjacency[idx] = neighbors
    end
    return adjacency
end

# Similarly for RegularGridQuadtree
function compute_adjacency(manifold, tree::QuadtreeCursor{<:RegularGridQuadtree})
    # Same index arithmetic approach
end
```

## File Organization

```
src/
  regridder/
    regridder.jl              # Regridder struct, method types, outer constructor
    regrid.jl                 # regrid! function (unchanged)
    intersection_areas.jl     # existing intersection area computation
    adjacency.jl              # compute_adjacency with dispatch
    gradients.jl              # Green's theorem gradient computation
    methods/
      conservative_1st.jl     # Regridder(manifold, ::Conservative1stOrder, ...)
      conservative_2nd.jl     # Regridder(manifold, ::Conservative2ndOrder, ...)
      nearest_neighbor.jl     # future
```

## 2nd Order Construction Algorithm

`Regridder(manifold, ::Conservative2ndOrder, dst, src; ...)`:

1. **Treeify grids** — Same as 1st order
2. **Compute source adjacency** — Dispatch to fast path for structured grids
3. **Compute source centroids** — `GO.centroid(manifold, cell)` for each source cell
4. **Compute gradient coefficients** — Green's theorem on neighbor polygon
   - Skip cells with < 3 neighbors (fallback to 1st order)
   - Validate centroid inside neighbor polygon
5. **Find candidate pairs** — Dual DFS between dst and src trees (same as 1st order)
6. **Compute weights** — For each overlap:
   - Compute overlap area and centroid
   - Compute `diff_cntr = overlap_centroid - src_centroid`
   - Add source weight: `(area/dst_area) - dot(diff_cntr, src_grad) * (area/dst_area)`
   - Add neighbor weights: `dot(diff_cntr, nbr_grad) * (area/dst_area)`
7. **Assemble sparse matrix** — Same structure as 1st order, but denser
8. **Compute areas and normalize** — Same as 1st order

## Backwards Compatibility

- Default method is `Conservative1stOrder()`
- Existing code calling `Regridder(dst, src)` continues to work unchanged
- The `regrid!` function signature is unchanged

## References

- ESMF implementation: `src/Infrastructure/Mesh/src/Regridding/ESMCI_Conserve2ndInterp.C`
- ESMF documentation: `src/Infrastructure/Regrid/doc/Regrid_implnotes.tex` (equations 286-330)
