# ConservativeRegridding.jl

[![Build Status](https://github.com/JuliaGEO/ConservativeRegridding.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaGEO/ConservativeRegridding.jl/actions/workflows/CI.yml?query=branch%3Amain)

ConservativeRegridding.jl provides functionality to regrid between two grids.
A grid is a tessellation of a space into polygons (or grid cells) each with an associated value.
Data on a grid is then referred to as _field_ whereas the grid itself just defines
that tessellation. Regridding is performed conservatively, meaning the area-weighted mean is preserved.
This is achieved by computing the intersection areas of all combinations between grid cells
of one grid and the other. All those intersections then provide the respective weights to
average from neighbouring cells.

## Usage

Let `grid2` (destination), `grid1` (source) be two arrays describing the polygon vertices on each grid.
Generally this is a vector of polygons (a vector of a vector of coordinate tuples) but this may also be
arranged as a matrix, with each column being one polygon, then

```julia
using ConservativeRegridding
R = ConservativeRegridding.Regridder(grid2, grid1)      # grid1 is source, grid2 the destination
```

And two fields `field2` (destination), `field1` (source) of data on these grids (normally a vector
unravelling the grid cells in the same order as the polygons are defined in `grid1`, `grid2` above)
are then regridded via

```julia
ConservativeRegridding.regrid!(field2, R, field1)
```

the re-use the regridder `R` for interpolation in the other direction (from `field2` to `field1`)
use

```julia
ConservativeRegridding.regrid!(field1, tranpose(R), field2)
```

Note that `R, tranpose(R)` share the same underlying data, `tranpose(R)` just creates a
new regridder based on a matrix tranpose and by swapping the area vectors, see below.

## Example

Let us tessellate the `[0, 2] x [0, 2]` space in two ways, blue grid and orange grid.

<img width="537" alt="Image" src="https://github.com/user-attachments/assets/04bc4b30-e8d3-4418-a53a-f6a4307d9bba" />

These grids are defined through their polygons, namely

```julia
# blue grid
blue_grid = [[(1, 1), (1, 0), (0, 0), (0, 1), (1, 1)],    # polygon 1
             [(2, 1), (2, 0), (1, 0), (1, 1), (2, 1)],    # polygon 2
             [(1, 2), (1, 1), (0, 1), (0, 2), (1, 2)],    # polygon 3
             [(2, 2), (2, 1), (1, 1), (1, 2), (2, 2)]]    # polygon 4

# orange grid
orange_grid = [[(0, 2), (1, 2), (0, 1), (0, 2)],            # polygon 1
               [(1, 2), (2, 2), (2, 1), (1, 2)],            # polygon 2
               [(0, 1), (1, 2), (2, 1), (1, 0), (0, 1)],    # polygon 3
               [(0, 0), (0, 1), (1, 0), (0, 0)],            # polygon 4
               [(1, 0), (2, 1), (2, 0), (1, 0)]]            # polygon 5
```

Note how all polygons are closed by repeating the first vertex as the last.
The regridder from 5-element orange grid to 4-element blue grid is now

```julia
julia> R = ConservativeRegridding.Regridder(blue_grid, orange_grid, normalize=false)
4×5 Regridder{SparseArrays.SparseMatrixCSC{Float64, Int64}, Vector{Float64}}
  ⋅    ⋅   0.5  0.5   ⋅ 
  ⋅    ⋅   0.5   ⋅   0.5
 0.5   ⋅   0.5   ⋅    ⋅ 
  ⋅   0.5  0.5   ⋅    ⋅ 

Source areas: [0.5, 0.5, 2.0, 0.5, 0.5]
Dest.  areas: [1.0, 1.0, 1.0, 1.0]
```

We disabled the area normalization with `normalize=false` as the units of the
areas are technically irrelevant, but here it shows nicely how every intersect
has the area of 1/2.

Start with some data on orange, allocate blue, and regrid

```julia
orange = rand(5)
blue = similar(orange, 4)
ConservativeRegridding.regrid!(blue, R, orange)
```

Now we test whether the mean is conserved

```julia
julia> sum(orange .* R.src_areas) / sum(R.src_areas)
0.4120316695084635

julia> sum(blue .* R.dst_areas) / sum(R.dst_areas)
0.4120316695084635
```

Indeed, even exactly. In this case with two very simple grids and intersects
there is not even rounding error which however has to be expected in general.

The backwards regridding via `tranpose(::Regridder)` is

```julia
julia> orange_tilde = similar(orange)
julia> ConservativeRegridding.regrid!(orange_tilde, transpose(R), blue)
5-element Vector{Float64}:
 0.48307071648257566
 0.49377895828829815
 0.4120316695084635
 0.30573011277148604
 0.3655468904914942

julia> orange
5-element Vector{Float64}:
 0.9108833154485086
 0.9322997990599536
 0.05525811751664267
 0.5562021080263294
 0.6758356634663457
```

still conserves the mean but has clearly lost in variance.

## Mathematical background

Let matrix $A$ be the intersection areas between the respective grids of the fields $d$
(destination) and $s$ (source), both unravelled into vectors, and $a_s$ and $a_d$ the areas
of the source and destination grid cells, then $s$ conservatively regridded to $d$ via
```math
d = (A s) / a_d
```
After the matrix-vector multiplication $As$ values are multiplied by the area of the grid cell
(the unit of entries in $A$) such that $/ a_s$ is an element-wise normalization by that area
to reobtained the regridded value. The areas $a_s$ and $a_d$ follow from row and column-wise
sums of $A$.

```math
\begin{aligned}
a_s &= \sum_i A_{ij} \\
a_d &= \sum_j A_{ij}
\end{aligned}
```

While the division with $a_s$ could be absorbed in $A$, not doing so means that $A$
can be reused to regrid in the other direction, from $d$ to $s$.

```math
\tilde{s} = (A^T d) / a_d
```

where the tilde was only added to highlight that conservative regridding is not a
perfectly invertible operation, while the mean is conserved, the variance usually is not.
With the equations above $A$ (and $a_s, a_d$ consequently) only has to be precomputed once
given two grids facilitating forward and backward regridding. 

Conservative regridding means that the area-weighted means are conserved

```math
\frac{\sum_i d_ia_{d, i}}{\sum_i a_{d, i}} = \frac{\sum_j s_j a_{s, j}}{\sum_j a_{d, j}}
```

And in many cases the two grids cover the same area, such that $\sum_i a_{d, i} = \sum_j a_{s, j}$,
e.g. the surface of a sphere.
