# 2nd Order Conservative Regridding Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 2nd order conservative regridding that uses gradient information for improved accuracy.

**Architecture:** Method types (`Conservative1stOrder`, `Conservative2ndOrder`) dispatch to different constructors. The Regridder struct gains a method type parameter. 2nd order computes adjacency, centroids, and gradient coefficients via Green's theorem, then builds an expanded sparse weight matrix.

**Tech Stack:** Julia, GeometryOps, SparseArrays, LinearAlgebra

---

## Task 1: Add Method Type Hierarchy

**Files:**
- Modify: `src/regridder/regridder.jl:1-25`

**Step 1: Write test for method types**

Create test file `test/methods.jl`:

```julia
using ConservativeRegridding
using Test

@testset "Method types" begin
    @test Conservative1stOrder() isa ConservativeRegridding.AbstractRegridMethod
    @test Conservative2ndOrder() isa ConservativeRegridding.AbstractRegridMethod
end
```

**Step 2: Run test to verify it fails**

Run: `julia --project=test -e 'include("test/methods.jl")'`
Expected: FAIL with "Conservative1stOrder not defined"

**Step 3: Add method types to regridder.jl**

Add after line 10 (after the constants):

```julia
"""
    AbstractRegridMethod

Abstract supertype for regridding methods.
Subtypes control how weights are computed during Regridder construction.
"""
abstract type AbstractRegridMethod end

"""
    Conservative1stOrder()

First-order conservative regridding. Weights are pure area ratios.
This is the default method and matches the original behavior.
"""
struct Conservative1stOrder <: AbstractRegridMethod end

"""
    Conservative2ndOrder()

Second-order conservative regridding. Weights incorporate gradient
information from neighboring cells for improved accuracy on smooth fields.

Note: Cannot be transposed. Build separate regridders for each direction.
"""
struct Conservative2ndOrder <: AbstractRegridMethod end
```

**Step 4: Export the types in ConservativeRegridding.jl**

In `src/ConservativeRegridding.jl`, change line 31:

```julia
public Regridder, regrid, regrid!
public AbstractRegridMethod, Conservative1stOrder, Conservative2ndOrder
public areas
```

**Step 5: Run test to verify it passes**

Run: `julia --project=test -e 'include("test/methods.jl")'`
Expected: PASS

**Step 6: Run full test suite**

Run: `julia --project=test test/runtests.jl`
Expected: All 1140 tests pass

**Step 7: Commit**

```bash
git add src/regridder/regridder.jl src/ConservativeRegridding.jl test/methods.jl
git commit -m "feat: add AbstractRegridMethod type hierarchy"
```

---

## Task 2: Update Regridder Struct with Method Type Parameter

**Files:**
- Modify: `src/regridder/regridder.jl:24-62`

**Step 1: Write test for method in Regridder**

Add to `test/methods.jl`:

```julia
@testset "Regridder stores method" begin
    # Create simple test grids
    import GeometryOps as GO, GeoInterface as GI
    make_square(x, y) = GI.Polygon([GI.LinearRing([
        (x, y), (x+1.0, y), (x+1.0, y+1.0), (x, y+1.0), (x, y)
    ])])

    dst = [make_square(0.0, 0.0), make_square(1.0, 0.0)]
    src = [make_square(0.0, 0.0), make_square(0.5, 0.0), make_square(1.0, 0.0)]

    R = ConservativeRegridding.Regridder(dst, src)
    @test R.method isa Conservative1stOrder
end
```

**Step 2: Run test to verify it fails**

Run: `julia --project=test -e 'include("test/methods.jl")'`
Expected: FAIL with "type Regridder has no field method"

**Step 3: Update Regridder struct**

Replace the struct definition (lines 24-36):

```julia
# Primary `Regridder` struct.
struct Regridder{M <: AbstractRegridMethod, W, A, V} <: AbstractRegridder
    "The regridding method used"
    method::M
    "Matrix of area intersections between cells on the source and destination grid"
    intersections::W
    "Vector of areas on the destination grid"
    dst_areas::A
    "Vector of areas on the source grid"
    src_areas::A
    "Dense vectors used as work-arrays if trying to regrid non-contiguous memory"
    dst_temp::V
    "Dense vectors used as work-arrays if trying to regrid non-contiguous memory"
    src_temp::V
end
```

**Step 4: Update Base.show**

Replace (around line 38):

```julia
function Base.show(io::IO, regridder::Regridder{M, W, A, V}) where {M, W, A, V}
    n2, n1 = size(regridder)
    println(io, "$n2×$n1 Regridder{$M, $W, $A, $V}")
    Base.print_array(io, regridder.intersections)
    println(io, "\n\nSource areas: ", regridder.src_areas)
    print(io, "Dest.  areas: ", regridder.dst_areas)
end
```

**Step 5: Update transpose**

Replace (around line 46):

```julia
"""$(TYPEDSIGNATURES)
Return a Regridder for the backwards regridding, i.e. from destination to source grid.
Does not copy any data, i.e. regridder for forward and backward share the same underlying arrays.

Note: Throws an error for Conservative2ndOrder since 2nd order weights are asymmetric.
"""
function LinearAlgebra.transpose(regridder::Regridder{<:Conservative2ndOrder})
    error("Cannot transpose a 2nd order regridder. " *
          "Build a separate Regridder(src, dst; method=Conservative2ndOrder()) for reverse direction.")
end

function LinearAlgebra.transpose(regridder::Regridder{M}) where M <: AbstractRegridMethod
    Regridder(regridder.method, transpose(regridder.intersections),
              regridder.src_areas, regridder.dst_areas,
              regridder.src_temp, regridder.dst_temp)
end
```

**Step 6: Update normalize!**

Replace (around line 54):

```julia
function LinearAlgebra.normalize!(regridder::Regridder)
    (; intersections) = regridder
    norm = maximum(intersections)
    intersections ./= norm
    regridder.src_areas ./= norm
    regridder.dst_areas ./= norm
    return regridder
end
```

**Step 7: Update constructor to include method**

Find the constructor around line 164 and update to pass method:

```julia
    # Construct the regridder.  Normalize if requested.
    regridder = Regridder(Conservative1stOrder(), intersections, dst_areas, src_areas, dst_temp, src_temp)
    normalize && LinearAlgebra.normalize!(regridder)

    return regridder
end
```

**Step 8: Run test to verify it passes**

Run: `julia --project=test -e 'include("test/methods.jl")'`
Expected: PASS

**Step 9: Run full test suite**

Run: `julia --project=test test/runtests.jl`
Expected: All tests pass

**Step 10: Commit**

```bash
git add src/regridder/regridder.jl test/methods.jl
git commit -m "feat: add method type parameter to Regridder struct"
```

---

## Task 3: Add Method Keyword to Constructor

**Files:**
- Modify: `src/regridder/regridder.jl:105-168`

**Step 1: Write test for method keyword**

Add to `test/methods.jl`:

```julia
@testset "Method keyword argument" begin
    import GeometryOps as GO, GeoInterface as GI
    make_square(x, y) = GI.Polygon([GI.LinearRing([
        (x, y), (x+1.0, y), (x+1.0, y+1.0), (x, y+1.0), (x, y)
    ])])

    dst = [make_square(0.0, 0.0), make_square(1.0, 0.0)]
    src = [make_square(0.0, 0.0), make_square(0.5, 0.0), make_square(1.0, 0.0)]

    # Default should be Conservative1stOrder
    R1 = ConservativeRegridding.Regridder(dst, src)
    @test R1.method isa Conservative1stOrder

    # Explicit Conservative1stOrder
    R2 = ConservativeRegridding.Regridder(dst, src; method=Conservative1stOrder())
    @test R2.method isa Conservative1stOrder

    # Results should be identical
    @test R1.intersections == R2.intersections
end
```

**Step 2: Run test to verify it fails**

Run: `julia --project=test -e 'include("test/methods.jl")'`
Expected: FAIL with "method not a valid keyword argument"

**Step 3: Update outer constructor**

Replace the outer constructor (around line 105):

```julia
function Regridder(dst, src; method::AbstractRegridMethod = Conservative1stOrder(), kwargs...)
    dst_manifold = GOCore.best_manifold(dst)
    src_manifold = GOCore.best_manifold(src)

    manifold = if dst_manifold != src_manifold
        # Implicitly promote to spherical
        if dst_manifold == GO.Planar() && src_manifold == GO.Spherical()
            GO.Spherical()
        elseif dst_manifold == GO.Spherical() && src_manifold == GO.Planar()
            GO.Spherical()
        else
            error("Destination and source manifolds must be the same.  Got $dst_manifold and $src_manifold.")
        end
    else
        dst_manifold
    end

    return Regridder(manifold, method, dst, src; kwargs...)
end
```

**Step 4: Update inner constructor signature**

Replace the inner constructor (around line 130):

```julia
function Regridder(
        manifold::M, method::Conservative1stOrder, dst, src;
        normalize = true,
        intersection_operator::F = DefaultIntersectionOperator(manifold),
        threaded = _default_threaded(manifold),
        kwargs...
    ) where {M <: Manifold, F}
    # "Normalize" the destination and source grids into trees.
    dst_tree = Trees.treeify(manifold, dst)
    src_tree = Trees.treeify(manifold, src)

    _threaded = booltype(threaded)

    # Compute the intersection areas.
    intersections = intersection_areas(
        manifold,
        _threaded,
        dst_tree, src_tree;
        intersection_operator,
        kwargs...
    )

    # Compute the areas of each cell
    # of the destination and source grids.
    dst_areas = areas(manifold, dst, dst_tree)
    src_areas = areas(manifold, src, src_tree)

    # Allocate temporary arrays for the regridding operation
    dst_temp = zeros(length(dst_areas))
    src_temp = zeros(length(src_areas))

    # Construct the regridder.  Normalize if requested.
    regridder = Regridder(method, intersections, dst_areas, src_areas, dst_temp, src_temp)
    normalize && LinearAlgebra.normalize!(regridder)

    return regridder
end
```

**Step 5: Run test to verify it passes**

Run: `julia --project=test -e 'include("test/methods.jl")'`
Expected: PASS

**Step 6: Run full test suite**

Run: `julia --project=test test/runtests.jl`
Expected: All tests pass

**Step 7: Commit**

```bash
git add src/regridder/regridder.jl test/methods.jl
git commit -m "feat: add method keyword argument to Regridder constructor"
```

---

## Task 4: Create Adjacency Module

**Files:**
- Create: `src/regridder/adjacency.jl`
- Modify: `src/ConservativeRegridding.jl`

**Step 1: Write test for adjacency computation**

Create `test/adjacency.jl`:

```julia
using ConservativeRegridding
using ConservativeRegridding: compute_adjacency
using Test
import GeometryOps as GO

@testset "Adjacency computation" begin
    @testset "Structured grid (3x3)" begin
        # Create a 3x3 grid of points (2x2 cells)
        points = [(Float64(i), Float64(j)) for i in 0:2, j in 0:2]
        grid = ConservativeRegridding.CellBasedGrid(GO.Planar(), points)
        tree = ConservativeRegridding.TopDownQuadtreeCursor(grid)

        adj = compute_adjacency(GO.Planar(), tree)

        # Cell layout (linear indices):
        # 2 4
        # 1 3
        # Cell 1 (0,0): neighbors are 2, 3, 4 (8-connectivity)
        @test sort(adj[1]) == [2, 3, 4]
        # Cell 2 (1,0): neighbors are 1, 3, 4
        @test sort(adj[2]) == [1, 3, 4]
        # Cell 3 (0,1): neighbors are 1, 2, 4
        @test sort(adj[3]) == [1, 2, 4]
        # Cell 4 (1,1): neighbors are 1, 2, 3
        @test sort(adj[4]) == [1, 2, 3]
    end

    @testset "Structured grid (4x4 points = 3x3 cells)" begin
        points = [(Float64(i), Float64(j)) for i in 0:3, j in 0:3]
        grid = ConservativeRegridding.CellBasedGrid(GO.Planar(), points)
        tree = ConservativeRegridding.TopDownQuadtreeCursor(grid)

        adj = compute_adjacency(GO.Planar(), tree)

        # Center cell (index 5 in column-major: i=2, j=2) should have 8 neighbors
        # Linear index for (2,2): 2 + (2-1)*3 = 5
        @test length(adj[5]) == 8

        # Corner cell (1,1) -> index 1 should have 3 neighbors
        @test length(adj[1]) == 3

        # Edge cell (2,1) -> index 2 should have 5 neighbors
        @test length(adj[2]) == 5
    end
end
```

**Step 2: Run test to verify it fails**

Run: `julia --project=test -e 'include("test/adjacency.jl")'`
Expected: FAIL with "compute_adjacency not defined"

**Step 3: Create adjacency.jl**

Create `src/regridder/adjacency.jl`:

```julia
#=
# Adjacency Computation

Compute cell adjacency for gradient computation in 2nd order regridding.
Neighbors are cells sharing edges or vertices (8-connectivity for quad grids).

Multiple dispatch provides fast paths for structured grids.
=#

"""
    compute_adjacency(manifold, tree) -> Vector{Vector{Int}}

Compute adjacency list for all cells in the tree.
Returns a vector where `adjacency[i]` contains the linear indices of all neighbors of cell `i`.

For structured grids (CellBasedGrid, RegularGrid), uses fast index arithmetic.
For unstructured grids, falls back to spatial queries.
"""
function compute_adjacency end

# Fast path for structured grids: CellBasedGrid
function compute_adjacency(manifold::GOCore.Manifold, tree::Trees.TopDownQuadtreeCursor{<:Trees.CellBasedGrid})
    grid = Trees.getgrid(tree)
    return _compute_structured_adjacency(grid)
end

# Fast path for structured grids: RegularGrid
function compute_adjacency(manifold::GOCore.Manifold, tree::Trees.TopDownQuadtreeCursor{<:Trees.RegularGrid})
    grid = Trees.getgrid(tree)
    return _compute_structured_adjacency(grid)
end

# Fast path for structured grids: ExplicitPolygonGrid
function compute_adjacency(manifold::GOCore.Manifold, tree::Trees.TopDownQuadtreeCursor{<:Trees.ExplicitPolygonGrid})
    grid = Trees.getgrid(tree)
    return _compute_structured_adjacency(grid)
end

"""
    _compute_structured_adjacency(grid) -> Vector{Vector{Int}}

Compute 8-connectivity adjacency for a structured grid using index arithmetic.
"""
function _compute_structured_adjacency(grid::Trees.AbstractCurvilinearGrid)
    ni = Trees.ncells(grid, 1)
    nj = Trees.ncells(grid, 2)
    n_total = ni * nj

    adjacency = Vector{Vector{Int}}(undef, n_total)

    for j in 1:nj, i in 1:ni
        idx = i + (j - 1) * ni  # column-major linear index
        neighbors = Int[]

        # 8-connectivity: all adjacent cells including diagonals
        for dj in -1:1, di in -1:1
            (di == 0 && dj == 0) && continue
            ni_new, nj_new = i + di, j + dj
            if 1 <= ni_new <= ni && 1 <= nj_new <= nj
                neighbor_idx = ni_new + (nj_new - 1) * ni
                push!(neighbors, neighbor_idx)
            end
        end

        adjacency[idx] = neighbors
    end

    return adjacency
end

# Generic fallback for unstructured grids (spatial queries)
# TODO: Implement using dual DFS to find touching polygons
function compute_adjacency(manifold::GOCore.Manifold, tree)
    error("compute_adjacency not yet implemented for unstructured grids ($(typeof(tree))). " *
          "Use a structured grid (CellBasedGrid, RegularGrid) for 2nd order regridding.")
end
```

**Step 4: Include adjacency.jl in ConservativeRegridding.jl**

In `src/ConservativeRegridding.jl`, add after line 29:

```julia
include("regridder/adjacency.jl")
```

**Step 5: Run test to verify it passes**

Run: `julia --project=test -e 'include("test/adjacency.jl")'`
Expected: PASS

**Step 6: Run full test suite**

Run: `julia --project=test test/runtests.jl`
Expected: All tests pass

**Step 7: Commit**

```bash
git add src/regridder/adjacency.jl src/ConservativeRegridding.jl test/adjacency.jl
git commit -m "feat: add adjacency computation for structured grids"
```

---

## Task 5: Create Gradient Computation Module

**Files:**
- Create: `src/regridder/gradients.jl`
- Modify: `src/ConservativeRegridding.jl`

**Step 1: Write test for gradient computation**

Create `test/gradients.jl`:

```julia
using ConservativeRegridding
using ConservativeRegridding: compute_gradient_coefficients, GradientInfo
using Test
import GeometryOps as GO

@testset "Gradient computation" begin
    @testset "Simple 3x3 grid" begin
        # Create a 4x4 grid of points (3x3 cells)
        points = [(Float64(i), Float64(j)) for i in 0:3, j in 0:3]
        grid = ConservativeRegridding.CellBasedGrid(GO.Planar(), points)
        tree = ConservativeRegridding.TopDownQuadtreeCursor(grid)

        grad_info = compute_gradient_coefficients(GO.Planar(), tree)

        # Should have gradient info for each cell
        @test length(grad_info) == 9

        # Center cell (index 5) should have valid gradients (8 neighbors)
        @test grad_info[5].valid
        @test length(grad_info[5].neighbor_indices) == 8

        # Corner cell (index 1) has only 3 neighbors, still valid (>= 3)
        @test grad_info[1].valid
        @test length(grad_info[1].neighbor_indices) == 3
    end

    @testset "2x2 grid - edge case" begin
        # 3x3 points = 2x2 cells
        points = [(Float64(i), Float64(j)) for i in 0:2, j in 0:2]
        grid = ConservativeRegridding.CellBasedGrid(GO.Planar(), points)
        tree = ConservativeRegridding.TopDownQuadtreeCursor(grid)

        grad_info = compute_gradient_coefficients(GO.Planar(), tree)

        # All cells have exactly 3 neighbors, so all should be valid
        for i in 1:4
            @test grad_info[i].valid
            @test length(grad_info[i].neighbor_indices) == 3
        end
    end
end
```

**Step 2: Run test to verify it fails**

Run: `julia --project=test -e 'include("test/gradients.jl")'`
Expected: FAIL with "compute_gradient_coefficients not defined"

**Step 3: Create gradients.jl**

Create `src/regridder/gradients.jl`:

```julia
#=
# Gradient Computation via Green's Theorem

Compute gradient coefficients for 2nd order conservative regridding.
Uses Green's theorem on the "neighbor polygon" formed by neighbor centroids.
=#

import GeometryOps as GO
import GeoInterface as GI
import GeometryOpsCore as GOCore
import LinearAlgebra

"""
    GradientInfo

Stores gradient coefficients for a single source cell.

Fields:
- `valid`: Whether gradients are valid (≥3 neighbors, centroid inside neighbor polygon)
- `centroid`: Cell centroid (x, y) or (x, y, z) for spherical
- `src_grad`: Gradient coefficient for the source cell itself
- `neighbor_indices`: Linear indices of neighbor cells
- `neighbor_grads`: Gradient coefficients for each neighbor
"""
struct GradientInfo{T}
    valid::Bool
    centroid::NTuple{2, T}
    src_grad::NTuple{2, T}
    neighbor_indices::Vector{Int}
    neighbor_grads::Vector{NTuple{2, T}}
end

# Constructor for invalid gradient info (fallback to 1st order)
function GradientInfo{T}(centroid::NTuple{2, T}, neighbor_indices::Vector{Int}) where T
    zero_grad = (zero(T), zero(T))
    GradientInfo{T}(false, centroid, zero_grad, neighbor_indices, [zero_grad for _ in neighbor_indices])
end

"""
    compute_gradient_coefficients(manifold, tree) -> Vector{GradientInfo}

Compute gradient coefficients for all cells using Green's theorem.

For each cell:
1. Get neighbors from adjacency
2. Compute neighbor centroids
3. Sort neighbors counter-clockwise around source centroid
4. Form neighbor polygon and apply Green's theorem
5. If validation fails, mark as invalid (will use 1st order fallback)
"""
function compute_gradient_coefficients(manifold::GOCore.Manifold, tree)
    adjacency = compute_adjacency(manifold, tree)
    n_cells = length(adjacency)
    T = Float64

    # Compute all centroids first
    centroids = Vector{NTuple{2, T}}(undef, n_cells)
    for i in 1:n_cells
        cell = Trees.getcell(tree, i)
        c = GO.centroid(manifold, cell)
        centroids[i] = (GI.x(c), GI.y(c))
    end

    # Compute gradient info for each cell
    grad_info = Vector{GradientInfo{T}}(undef, n_cells)

    for i in 1:n_cells
        neighbor_indices = adjacency[i]
        src_centroid = centroids[i]

        # Need at least 3 neighbors for valid gradient
        if length(neighbor_indices) < 3
            grad_info[i] = GradientInfo{T}(src_centroid, neighbor_indices)
            continue
        end

        # Get neighbor centroids
        neighbor_centroids = [centroids[j] for j in neighbor_indices]

        # Sort neighbors counter-clockwise around source centroid
        sorted_indices, sorted_centroids = _sort_neighbors_ccw(src_centroid, neighbor_indices, neighbor_centroids)

        # Compute gradient coefficients via Green's theorem
        src_grad, neighbor_grads, valid = _compute_greens_theorem_gradients(
            src_centroid, sorted_centroids
        )

        grad_info[i] = GradientInfo{T}(
            valid,
            src_centroid,
            src_grad,
            sorted_indices,
            neighbor_grads
        )
    end

    return grad_info
end

"""
Sort neighbor indices and centroids counter-clockwise around the source centroid.
"""
function _sort_neighbors_ccw(src_centroid::NTuple{2, T}, indices::Vector{Int},
                              centroids::Vector{NTuple{2, T}}) where T
    # Compute angles from source centroid to each neighbor
    angles = [atan(c[2] - src_centroid[2], c[1] - src_centroid[1]) for c in centroids]

    # Sort by angle
    perm = sortperm(angles)

    return indices[perm], centroids[perm]
end

"""
Compute gradient coefficients using Green's theorem on the neighbor polygon.

Returns (src_grad, neighbor_grads, valid).
"""
function _compute_greens_theorem_gradients(src_centroid::NTuple{2, T},
                                            neighbor_centroids::Vector{NTuple{2, T}}) where T
    n = length(neighbor_centroids)

    # Initialize gradients
    src_grad = (zero(T), zero(T))
    neighbor_grads = Vector{NTuple{2, T}}(undef, n)

    # Compute area of neighbor polygon (for validation and normalization)
    area = _polygon_area(neighbor_centroids)

    if abs(area) < eps(T) * 100
        # Degenerate polygon
        return src_grad, [src_grad for _ in 1:n], false
    end

    # Check if source centroid is inside neighbor polygon
    if !_point_in_polygon(src_centroid, neighbor_centroids)
        return src_grad, [src_grad for _ in 1:n], false
    end

    # Apply Green's theorem
    # For each edge of the neighbor polygon, compute contribution to gradient
    inv_area = one(T) / area

    for i in 1:n
        j = mod1(i + 1, n)  # next vertex (wrapping)

        p1 = neighbor_centroids[i]
        p2 = neighbor_centroids[j]

        # Edge vector
        dx = p2[1] - p1[1]
        dy = p2[2] - p1[2]

        # Outward normal (perpendicular to edge, pointing outward for CCW polygon)
        # For a CCW polygon, outward normal is (dy, -dx)
        nx = dy
        ny = -dx

        # Edge length
        edge_len = sqrt(dx^2 + dy^2)
        if edge_len < eps(T)
            continue
        end

        # Normalize and scale by edge length / (2 * area)
        # This is the Green's theorem coefficient
        scale = edge_len * inv_area / 2

        # Gradient contribution for vertex i and j
        grad_contrib = (nx * scale, ny * scale)

        # Add to neighbor gradients (each edge contributes to both endpoints)
        if i == 1
            neighbor_grads[i] = grad_contrib
        else
            neighbor_grads[i] = (neighbor_grads[i][1] + grad_contrib[1],
                                 neighbor_grads[i][2] + grad_contrib[2])
        end

        if j == 1
            neighbor_grads[j] = (neighbor_grads[j][1] + grad_contrib[1],
                                 neighbor_grads[j][2] + grad_contrib[2])
        elseif j > i  # Only initialize if not already set
            neighbor_grads[j] = grad_contrib
        else
            neighbor_grads[j] = (neighbor_grads[j][1] + grad_contrib[1],
                                 neighbor_grads[j][2] + grad_contrib[2])
        end
    end

    # Source gradient is negative sum of neighbor gradients (conservation)
    src_grad_x = zero(T)
    src_grad_y = zero(T)
    for ng in neighbor_grads
        src_grad_x -= ng[1]
        src_grad_y -= ng[2]
    end
    src_grad = (src_grad_x, src_grad_y)

    return src_grad, neighbor_grads, true
end

"""
Compute signed area of a polygon given as a list of vertices (CCW = positive).
"""
function _polygon_area(vertices::Vector{NTuple{2, T}}) where T
    n = length(vertices)
    area = zero(T)
    for i in 1:n
        j = mod1(i + 1, n)
        area += vertices[i][1] * vertices[j][2]
        area -= vertices[j][1] * vertices[i][2]
    end
    return area / 2
end

"""
Check if a point is inside a polygon (ray casting algorithm).
"""
function _point_in_polygon(point::NTuple{2, T}, vertices::Vector{NTuple{2, T}}) where T
    n = length(vertices)
    inside = false

    px, py = point
    j = n

    for i in 1:n
        xi, yi = vertices[i]
        xj, yj = vertices[j]

        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi)
            inside = !inside
        end
        j = i
    end

    return inside
end
```

**Step 4: Include gradients.jl in ConservativeRegridding.jl**

In `src/ConservativeRegridding.jl`, add after the adjacency include:

```julia
include("regridder/gradients.jl")
```

**Step 5: Run test to verify it passes**

Run: `julia --project=test -e 'include("test/gradients.jl")'`
Expected: PASS

**Step 6: Run full test suite**

Run: `julia --project=test test/runtests.jl`
Expected: All tests pass

**Step 7: Commit**

```bash
git add src/regridder/gradients.jl src/ConservativeRegridding.jl test/gradients.jl
git commit -m "feat: add gradient computation via Green's theorem"
```

---

## Task 6: Implement Conservative2ndOrder Constructor

**Files:**
- Modify: `src/regridder/regridder.jl`

**Step 1: Write test for 2nd order regridder**

Add to `test/methods.jl`:

```julia
@testset "Conservative2ndOrder construction" begin
    import GeometryOps as GO

    # Create a 5x5 grid of points (4x4 cells) - need enough cells for gradients
    src_points = [(Float64(i), Float64(j)) for i in 0:4, j in 0:4]
    dst_points = [(Float64(i)*2, Float64(j)*2) for i in 0:2, j in 0:2]  # 2x2 cells, coarser

    R = ConservativeRegridding.Regridder(dst_points, src_points; method=Conservative2ndOrder())

    @test R.method isa Conservative2ndOrder
    @test size(R.intersections, 1) == 4  # 2x2 dst cells
    @test size(R.intersections, 2) == 16 # 4x4 src cells

    # 2nd order matrix should be denser than 1st order (has neighbor contributions)
    R1 = ConservativeRegridding.Regridder(dst_points, src_points; method=Conservative1stOrder())
    @test nnz(R.intersections) >= nnz(R1.intersections)
end

@testset "Conservative2ndOrder transpose error" begin
    import GeometryOps as GO

    src_points = [(Float64(i), Float64(j)) for i in 0:4, j in 0:4]
    dst_points = [(Float64(i)*2, Float64(j)*2) for i in 0:2, j in 0:2]

    R = ConservativeRegridding.Regridder(dst_points, src_points; method=Conservative2ndOrder())

    @test_throws ErrorException transpose(R)
end
```

**Step 2: Run test to verify it fails**

Run: `julia --project=test -e 'include("test/methods.jl")'`
Expected: FAIL with "no method matching Regridder(::Planar, ::Conservative2ndOrder, ...)"

**Step 3: Add Conservative2ndOrder constructor**

Add at the end of `src/regridder/regridder.jl`:

```julia
"""
Second-order conservative regridder constructor.

Computes weights that incorporate gradient information from neighboring cells.
"""
function Regridder(
        manifold::M, method::Conservative2ndOrder, dst, src;
        normalize = true,
        intersection_operator::F = DefaultIntersectionOperator(manifold),
        threaded = _default_threaded(manifold),
        kwargs...
    ) where {M <: Manifold, F}

    # Treeify grids
    dst_tree = Trees.treeify(manifold, dst)
    src_tree = Trees.treeify(manifold, src)

    _threaded = booltype(threaded)

    # Compute gradient coefficients for source grid
    grad_info = compute_gradient_coefficients(manifold, src_tree)

    # Compute source centroids (already in grad_info, extract for convenience)
    src_centroids = [gi.centroid for gi in grad_info]

    # Get candidate pairs via dual DFS
    predicate_f = if M <: Spherical
        GO.UnitSpherical._intersects
    else
        Extents.intersects
    end
    candidate_idxs = get_all_candidate_pairs(_threaded, predicate_f, src_tree, dst_tree)

    # Compute 2nd order weights
    n_dst = prod(Trees.ncells(dst_tree))
    n_src = prod(Trees.ncells(src_tree))

    i_dst = Int[]
    i_src = Int[]
    weights = Float64[]

    # Pre-allocate with estimate
    sizehint!(i_dst, length(candidate_idxs) * 5)  # ~5x for neighbor contributions
    sizehint!(i_src, length(candidate_idxs) * 5)
    sizehint!(weights, length(candidate_idxs) * 5)

    for (src_idx, dst_idx) in candidate_idxs
        src_poly = Trees.getcell(src_tree, src_idx)
        dst_poly = Trees.getcell(dst_tree, dst_idx)

        # Compute intersection
        intersection_polys = GO.intersection(
            GO.FosterHormannClipping(manifold), src_poly, dst_poly;
            target = GI.PolygonTrait()
        )

        overlap_area = GO.area(manifold, intersection_polys)
        if overlap_area <= 0
            continue
        end

        # Compute overlap centroid
        overlap_centroid = if !isempty(intersection_polys)
            c = GO.centroid(manifold, intersection_polys)
            (GI.x(c), GI.y(c))
        else
            continue
        end

        gi = grad_info[src_idx]
        src_centroid = gi.centroid

        # diff_cntr = overlap_centroid - src_centroid
        diff_cntr = (overlap_centroid[1] - src_centroid[1],
                     overlap_centroid[2] - src_centroid[2])

        if gi.valid
            # 2nd order: source weight with gradient correction
            src_grad = gi.src_grad
            grad_term = diff_cntr[1] * src_grad[1] + diff_cntr[2] * src_grad[2]
            src_weight = overlap_area - grad_term * overlap_area

            push!(i_dst, dst_idx)
            push!(i_src, src_idx)
            push!(weights, src_weight)

            # Neighbor contributions
            for (nbr_idx, nbr_grad) in zip(gi.neighbor_indices, gi.neighbor_grads)
                nbr_weight = (diff_cntr[1] * nbr_grad[1] + diff_cntr[2] * nbr_grad[2]) * overlap_area
                if abs(nbr_weight) > eps(Float64) * overlap_area
                    push!(i_dst, dst_idx)
                    push!(i_src, nbr_idx)
                    push!(weights, nbr_weight)
                end
            end
        else
            # Fallback to 1st order for this cell
            push!(i_dst, dst_idx)
            push!(i_src, src_idx)
            push!(weights, overlap_area)
        end
    end

    # Assemble sparse matrix
    intersections = SparseArrays.sparse(i_dst, i_src, weights, n_dst, n_src)

    # Compute areas
    dst_areas = areas(manifold, dst, dst_tree)
    src_areas = areas(manifold, src, src_tree)

    # Allocate temp arrays
    dst_temp = zeros(length(dst_areas))
    src_temp = zeros(length(src_areas))

    # Construct regridder
    regridder = Regridder(method, intersections, dst_areas, src_areas, dst_temp, src_temp)
    normalize && LinearAlgebra.normalize!(regridder)

    return regridder
end
```

**Step 4: Run test to verify it passes**

Run: `julia --project=test -e 'include("test/methods.jl")'`
Expected: PASS

**Step 5: Run full test suite**

Run: `julia --project=test test/runtests.jl`
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/regridder/regridder.jl test/methods.jl
git commit -m "feat: implement Conservative2ndOrder constructor"
```

---

## Task 7: Add Accuracy Test for 2nd Order

**Files:**
- Create: `test/accuracy.jl`

**Step 1: Write accuracy comparison test**

Create `test/accuracy.jl`:

```julia
using ConservativeRegridding
using Test
import GeometryOps as GO
using Statistics

@testset "2nd order accuracy improvement" begin
    # Create fine source grid (10x10 cells)
    src_points = [(Float64(i)/10, Float64(j)/10) for i in 0:10, j in 0:10]

    # Create coarser destination grid (5x5 cells)
    dst_points = [(Float64(i)/5, Float64(j)/5) for i in 0:5, j in 0:5]

    # Linear field: f(x,y) = 2x + 3y
    # 2nd order should be exact for linear fields
    src_field = Float64[]
    for j in 1:10, i in 1:10
        # Cell center
        x = (i - 0.5) / 10
        y = (j - 0.5) / 10
        push!(src_field, 2*x + 3*y)
    end

    # Expected values at destination cell centers
    expected = Float64[]
    for j in 1:5, i in 1:5
        x = (i - 0.5) / 5
        y = (j - 0.5) / 5
        push!(expected, 2*x + 3*y)
    end

    # Regrid with 1st and 2nd order
    R1 = ConservativeRegridding.Regridder(dst_points, src_points; method=Conservative1stOrder())
    R2 = ConservativeRegridding.Regridder(dst_points, src_points; method=Conservative2ndOrder())

    dst1 = zeros(25)
    dst2 = zeros(25)

    ConservativeRegridding.regrid!(dst1, R1, src_field)
    ConservativeRegridding.regrid!(dst2, R2, src_field)

    # Both should give reasonable results
    @test all(isfinite, dst1)
    @test all(isfinite, dst2)

    # 2nd order should be at least as accurate as 1st order
    error1 = mean(abs.(dst1 .- expected))
    error2 = mean(abs.(dst2 .- expected))

    @test error2 <= error1 + 1e-10  # Allow small numerical tolerance

    # For a linear field, 2nd order should be nearly exact
    @test error2 < 0.1 * error1 || error2 < 1e-10
end
```

**Step 2: Run test**

Run: `julia --project=test -e 'include("test/accuracy.jl")'`
Expected: PASS

**Step 3: Commit**

```bash
git add test/accuracy.jl
git commit -m "test: add accuracy comparison test for 2nd order regridding"
```

---

## Task 8: Add Tests to Main Test Suite

**Files:**
- Modify: `test/runtests.jl`

**Step 1: Add new test files to runtests.jl**

Read current runtests.jl and add the new test files:

```julia
@safetestset "Unit tests: Methods" begin include("methods.jl") end
@safetestset "Unit tests: Adjacency" begin include("adjacency.jl") end
@safetestset "Unit tests: Gradients" begin include("gradients.jl") end
@safetestset "Unit tests: 2nd Order Accuracy" begin include("accuracy.jl") end
```

**Step 2: Run full test suite**

Run: `julia --project=test test/runtests.jl`
Expected: All tests pass

**Step 3: Commit**

```bash
git add test/runtests.jl
git commit -m "test: add 2nd order regridding tests to main suite"
```

---

## Task 9: Final Verification and Cleanup

**Step 1: Run full test suite one more time**

Run: `julia --project=test test/runtests.jl`
Expected: All tests pass

**Step 2: Review all changes**

Run: `git log --oneline origin/main..HEAD`
Expected: See all commits from this implementation

**Step 3: Check for any TODO comments or incomplete implementations**

Run: `grep -r "TODO" src/`
Review any TODOs and decide if they need addressing now or can be deferred.

---

## Summary

This plan implements 2nd order conservative regridding in 9 tasks:

1. **Method types** - Add `AbstractRegridMethod` hierarchy
2. **Struct update** - Add method type parameter to `Regridder`
3. **Constructor keyword** - Add `method` kwarg with dispatch
4. **Adjacency** - Fast 8-connectivity for structured grids
5. **Gradients** - Green's theorem coefficient computation
6. **2nd order constructor** - Main implementation
7. **Accuracy test** - Verify 2nd order improves on 1st order
8. **Test integration** - Add to main test suite
9. **Final verification** - Ensure everything works together

Each task has explicit tests that must pass before moving on, ensuring incremental correctness.
