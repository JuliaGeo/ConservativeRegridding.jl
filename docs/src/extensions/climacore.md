# ClimaCore Spectral Element Extension

ConservativeRegridding.jl provides specialized regridding support for
[ClimaCore.jl](https://github.com/CliMA/ClimaCore.jl) spectral element (SE) spaces.
This extension is loaded automatically when `ClimaCore` is imported alongside
`ConservativeRegridding`.

## Background: Why SE ≠ FV

A finite volume (FV) grid stores one value per cell. A spectral element grid stores
values at ``N_q^2`` quadrature nodes per element, where the polynomial degree is
``N_q - 1``.

SE regridding approaches work directly with the nodal values and their
Jacobian integration weights ``W_{e,i,j}`` (obtained from
`ClimaCore.Spaces.weighted_jacobian`).  The key observation is that the SE
integral over an element is given by

```math
\int_e f \, dA \approx \sum_{i,j} W_{e,i,j} \, f_{e,i,j}
```

where ``e`` is a SE element, ``i,j`` are nodal indices within the element, and ``f`` is the function being evaluated on the SE space.

## Regridding Cases

### Case 1: SE → FV

For each FV cell ``k``, accumulate the Jacobian-weighted values of all SE nodes
that fall inside it, then normalize by the cell area:

```math
f^{\text{dst}}_k = \frac{1}{A_k} \sum_{\substack{(e,i,j) | \\ x_{e,i,j} \in k}} W_{e,i,j} \, f^{\text{src}}_{e,i,j}
```
where ``A_k`` is the area of FV cell ``k``, and ``x_{e,i,j}`` is the node at indices ``i,j`` within element ``e``.

This is implemented as a sparse matrix–vector multiply followed by element-wise
division by destination areas.  The regridder type for this case is
[`SEtoFVRegridder`](@ref ConservativeRegridding.SEtoFVRegridder).

### Case 2: FV → SE

Each SE node receives the value of the FV cell that contains it:

```math
f^{\text{dst}}_{e,i,j} = f^{\text{src}}_k \quad \text{where } x_{e,i,j} \in k
```

No area normalization is needed. The regridder type for this case is
[`FVtoSERegridder`](@ref ConservativeRegridding.FVtoSERegridder).

### Case 3: SE → SE

Each destination element receives a source-weighted integral from all source
nodes that fall inside it, normalized by the destination element area. All
``N_q^2`` nodes within a destination element share the same regridded value:

```math
f^{\text{dst}}_{e'} = \frac{1}{A_{e'}} \sum_{\substack{(e,i,j) \\ x_{e,i,j} \in e'}} W_{e,i,j} \, f^{\text{src}}_{e,i,j}
```

The regridder type for this case is
[`SEtoSERegridder`](@ref ConservativeRegridding.SEtoSERegridder).

## Resolution Mismatch Fallback

When source and destination grids have very different resolutions, a destination
cell may contain no source SE nodes.  In this case the extension falls back to
intersection-area weighted element averages for those cells, equivalent to the
standard FV approach.

## Usage

### Constructing a Regridder

The `Regridder` constructor automatically detects when one or both arguments are
a `ClimaCore.Spaces.AbstractSpectralElementSpace` and selects the appropriate
SE regridder type:

```julia
using ConservativeRegridding, ClimaCore, Oceananigans
using ClimaCore: CommonSpaces

# Create grids
space = CommonSpaces.CubedSphereSpace(; radius = 6.371e6, n_quad_points = 4, h_elem = 16)
grid  = LatitudeLongitudeGrid(size=(360, 180, 1), longitude=(0, 360), latitude=(-90, 90), z=(0,1))

# SE → FV
R_se2fv = Regridder(grid, space)      # returns SEtoFVRegridder

# FV → SE
R_fv2se = Regridder(space, grid)      # returns FVtoSERegridder

# SE → SE
space2 = CommonSpaces.CubedSphereSpace(; radius = 6.371e6, n_quad_points = 4, h_elem = 32)
R_se2se = Regridder(space2, space)    # returns SEtoSERegridder
```

### Converting between ClimaCore fields and flat vectors

The extension provides helpers to bridge ClimaCore's data layout with the flat
nodal vectors expected by `regrid!`:

```julia
const Ext = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingClimaCoreExt)

# ClimaCore field → flat vector
src_vec = Ext.se_field_to_vec(field)

# Flat vector → ClimaCore field
dst_field = Fields.zeros(space)
Ext.vec_to_se_field!(dst_field, dst_vec)

# Node positions and weights (for diagnostics / verification)
positions = Ext.se_node_positions(space)   # Vector{UnitSphericalPoint}
weights   = Ext.se_node_weights(space)     # Vector{Float64}
```

### Regridding

```julia
# Forward: SE → FV
fv_vals = zeros(360 * 180)
regrid!(fv_vals, R_se2fv, src_vec)

# Backward: FV → SE
se_vals = zeros(Nq^2 * Nh)
regrid!(se_vals, R_fv2se, fv_vals)

# SE → SE
se2_vals = zeros(Nq2^2 * Nh2)
regrid!(se2_vals, R_se2se, src_vec)
```

### Verifying Conservation

For the SE side, ClimaCore's `sum(field)` computes the proper quadrature integral
``\sum_{e,i,j} W_{e,i,j} f_{e,i,j}`` and should be used for conservation checks:

```julia
using ConservativeRegridding: areas, Trees
import GeometryOps as GO

# SE → FV conservation
fv_areas = areas(GO.Spherical(), Trees.treeify(grid))
@assert sum(fv_vals .* fv_areas) ≈ sum(src_field)   # ClimaCore sum on RHS
```
