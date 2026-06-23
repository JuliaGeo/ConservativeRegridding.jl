# ClimaCore Spectral Element Extension

ConservativeRegridding.jl provides conservative regridding between finite volume (FV)
grids and [ClimaCore.jl](https://github.com/CliMA/ClimaCore.jl) spectral element (SE)
spaces. The extension is loaded automatically when `ClimaCore` is imported alongside
`ConservativeRegridding`.

## Spectral element and finite volume integrals

A finite volume grid carries one mean value ``\bar f_i`` per cell, so the integral of a
field over cell ``i`` is

```math
\int_i f \, \mathrm{d}A = \bar f_i \, A_i ,
```

where ``A_i`` is the cell area. 

A spectral element space carries ``N_q^2`` nodal values
``f^e_{ij}`` per element ``e`` (polynomial degree ``N_q - 1``), expanded in a tensor
product of one-dimensional Lagrange basis polynomials,

```math
f^e(\xi, \eta) = \sum_{i,j} f^e_{ij} \, \phi_i(\xi) \, \phi_j(\eta) ,
```

over the reference square ``(\xi, \eta) \in [-1, 1]^2``. The integral over an element is
evaluated by Gauss–Legendre–Lobatto (GLL) quadrature,

```math
\int_e f \, \mathrm{d}A \approx \sum_{i,j} W^e_{ij} \, f^e_{ij} ,
\qquad
W^e_{ij} = w_i \, w_j \, J^e_{ij} ,
```

where ``w_i`` are the quadrature weights, ``J^e_{ij}`` the Jacobian of the
reference-to-physical map at node ``(i,j)``, and ``W^e_{ij}`` the *weighted Jacobian*
stored by `ClimaCore.Spaces.weighted_jacobian`.

## Implemented regridding directions

Three directions are conceivable; the extension implements the two that involve a finite
volume side:

| Direction | Method |
|-----------|--------|
| SE ``\to`` FV | area-weighted average of the SE field over each cell (see *SE → FV* below) |
| FV ``\to`` SE | per-element ``L^2`` projection onto the SE basis (see *FV → SE* below) |
| SE ``\to`` SE | *not supported* — compose SE ``\to`` FV ``\to`` SE |

Both implemented directions are built from the same family of weights ``B(k,(e,i,j))``,
defined and evaluated below.

## The intersection weights ``B``

For a finite volume cell ``k`` and a spectral element ``e``, define

```math
B(k, (e,i,j)) = \int_{k \cap e} \phi_i(\xi) \, \phi_j(\eta) \, J^e(\xi, \eta) \,
                \mathrm{d}\xi \, \mathrm{d}\eta ,
```

the integral of the SE basis function ``(i,j)`` over the intersection of cell ``k`` with
element ``e``. Because the reference-to-physical map has Jacobian determinant ``J^e``,
the physical area element is ``\mathrm{d}A = J^e \, \mathrm{d}\xi \, \mathrm{d}\eta``, and
the weight is equivalently a *physical-space* integral with no explicit Jacobian,

```math
B(k, (e,i,j)) = \int_{k \cap e} \phi_i\big(\xi(x)\big) \, \phi_j\big(\eta(x)\big) \,
                \mathrm{d}A .
```

The extension evaluates this physical-space form directly (`accumulate_principled_b`) by
(1) fan-triangulating the intersection polygon ``k \cap e`` (already computed by 
ConservativeRegridding.jl via GeometryOps.jl) from its centroid (2) using a barycentric 
Gauss rule on each triangle, and (3) evaluating the basis functions at each quadrature point
after inverting the element map to recover ``(\xi, \eta)``. The physical triangle areas
carry the Jacobian, so ``J^e`` does not appear explicitly. This integration is exact to
the quadrature order and is therefore high-order accurate in the element size.

Two identities follow from the definition and are used below. Since the cells ``k`` tile
the domain,

```math
\sum_k B(k, (e,i,j)) = W^e_{ij} ,
```

and since the elements ``e`` tile the domain (and ``\sum_{i,j} \phi_i \phi_j = 1``),

```math
\sum_{e,i,j} B(k, (e,i,j)) = A_k .
```

## SE → FV

The destination cell value is the area-weighted integral of the source SE field over the
cell,

```math
f^{\mathrm{dst}}_k = \frac{1}{A_k} \sum_{e,i,j} B(k,(e,i,j)) \, f^{\mathrm{src},e}_{ij} .
```

This is assembled as a sparse matrix with entries ``B(k,(e,i,j))`` (rows indexed by FV
cell, columns by SE node), and `regrid!` performs the matrix–vector product followed by
division by the destination areas. Conservation is exact in the continuous weights: summing
over ``k`` and using ``\sum_k B = W^e_{ij}`` recovers ``\sum_{e,i,j} W^e_{ij}
f^{\mathrm{src},e}_{ij}``, the SE quadrature integral of the source.

## FV → SE

The destination SE field is defined by the weak condition that it agree with the source in
every test direction of the SE space: find ``f^{\mathrm{dst}}`` such that

```math
\int f^{\mathrm{dst}} \, \psi \, \mathrm{d}A = \int f^{\mathrm{src}} \, \psi \, \mathrm{d}A
```

for all basis functions ``\psi`` of the SE space. Taking ``\psi = 1`` makes this
conservative. Expanding ``f^{\mathrm{dst}}`` in the SE basis turns the condition into a
linear system ``\mathbf{M} \, \mathbf{f}^{\mathrm{dst}} = \mathbf{b}`` with the mass matrix
``\mathbf{M}_{\alpha\beta} = \int \psi_\alpha \psi_\beta \, \mathrm{d}A``. Both sides are
block-diagonal over elements, so the problem reduces to a per-element solve. The local mass
matrix is

```math
M^e_{(ab),(cd)} = \int_e \phi_a(\xi) \, \phi_b(\eta) \, \phi_c(\xi) \, \phi_d(\eta) \,
                  J^e(\xi, \eta) \, \mathrm{d}\xi \, \mathrm{d}\eta
```

(assembled by `compute_local_mass_matrix`, with the index pairs flattened by
`ClimaCore.Utilities.linear_ind`). Its integrand has degree ``3(N_q - 1)`` per direction,
so it is integrated with a GLL rule of order ``n \ge (3N_q - 2)/2``; the Jacobian ``J^e``
is interpolated from its nodal values where needed (`element_jacobian_at`). The right-hand
side reuses the intersection weights,

```math
b^e_{ij} = \sum_k B(k,(e,i,j)) \, f^{\mathrm{src}}_k ,
```

and the per-element solve gives the assembled regridding operator,

```math
f^{\mathrm{dst},e}_{ij} = \sum_{a,b} \big[(M^e)^{-1}\big]_{(ij),(ab)} \,
                          \sum_k B(k,(e,a,b)) \, f^{\mathrm{src}}_k .
```

Using the full mass-matrix inverse — rather than dividing by the lumped diagonal
``W^e_{ij}`` — preserves constants exactly and is the optimal ``L^2`` fit of the source on
the SE basis over each element.

Two corrections appear in the implementation:

- **Quadrature-consistency rescaling.** The mass matrix is integrated by tensor-product GLL
  on the reference square, while ``B`` is integrated by triangulation of the physical
  polygon; the two cover slightly different domains. Before solving, each row of ``M^e`` is
  rescaled so its row sum matches the corresponding column sum of ``B``, which restores
  exact constant preservation.
- **Empty-overlap fallback.** A destination node may lie outside the source coverage (for
  example a tripolar ocean grid that stops well short of the South Pole regridded onto a
  full-sphere SE space), giving ``b^e_{ij} = 0``. Only the covered nodes enter the local
  solve; uncovered nodes are left untouched rather than forced to zero.

## SE → SE

Direct SE ``\to`` SE regridding is not supported: the `Regridder` constructor raises an
error for two SE arguments and directs the caller to compose the two implemented
directions, SE ``\to`` FV ``\to`` SE, with an intermediate finite volume grid.

## Usage

### Constructing a regridder

`Regridder` detects which argument is a `ClimaCore.Spaces.AbstractSpectralElementSpace` (or
a `ClimaCore.Fields.Field`) and selects the corresponding direction. The first argument is
the destination, the second the source.

Note that `Regridder`, `regrid!`, and `areas` are marked public but are *not* exported, so
`using ConservativeRegridding` does not bring them into scope; import them explicitly (as
below) or qualify them as `ConservativeRegridding.Regridder`, etc.

```julia
using ConservativeRegridding, ClimaCore, Oceananigans
using ConservativeRegridding: Regridder, regrid!
using ClimaCore: CommonSpaces

space = CommonSpaces.CubedSphereSpace(; radius = 6.371e6, n_quad_points = 4, h_elem = 16)
grid  = LatitudeLongitudeGrid(size = (360, 180, 1),
                              longitude = (0, 360), latitude = (-90, 90), z = (0, 1))

R_se2fv = Regridder(grid, space)   # SE source → FV destination
R_fv2se = Regridder(space, grid)   # FV source → SE destination
```

### Regridding

`regrid!` accepts and writes ClimaCore `Field`s directly on the SE side, flattening to and
from the nodal layout internally (and applying a weighted direct stiffness summation when
writing an SE destination):

```julia
using ClimaCore: Fields

# SE → FV
src_field = Fields.coordinate_field(space).lat
fv_vals   = zeros(360 * 180)
regrid!(fv_vals, R_se2fv, src_field)

# FV → SE
dst_field = Fields.zeros(space)
regrid!(dst_field, R_fv2se, fv_vals)
```

### Converting between ClimaCore fields and flat vectors

```julia
const ClimaCoreExt = Base.get_extension(ConservativeRegridding, :ConservativeRegriddingClimaCoreExt)

field = Fields.coordinate_field(space).lat           # an example SE field

src_vec = ClimaCoreExt.se_field_to_vec(field)        # Field → flat vector

dst_field = Fields.zeros(space)
ClimaCoreExt.vec_to_se_field!(dst_field, src_vec)    # flat vector → Field

positions = ClimaCoreExt.se_node_positions(space)    # Vector{UnitSphericalPoint}
weights   = ClimaCoreExt.se_node_weights(space)      # nodal weights W^e_{ij}
```

## Checking conservation

On the SE side, `ClimaCore.sum(field)` computes the quadrature integral ``\sum_{e,i,j}
W^e_{ij} f^e_{ij}`` and is the correct quantity for conservation checks. On the FV side the
integral is the area-weighted sum:

```julia
using ConservativeRegridding: areas, Trees
import GeometryOps as GO

fv_areas = areas(GO.Spherical(), Trees.treeify(grid))
@assert sum(fv_vals .* fv_areas) ≈ sum(src_field)   # SE → FV conservation
```
