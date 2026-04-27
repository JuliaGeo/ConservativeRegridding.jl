# XESMF / ESMF comparison test — plan

Date: 2026-04-26
Status: Ready to implement
Branch context: `as/newapi`
Target file: `test/xesmf_comparison.jl` (will replace the existing
`test/usecases/xesmf_comparison.jl` and be re-wired in `test/runtests.jl`)

## Goal

Validate ConservativeRegridding.jl's 1st- and 2nd-order conservative weights
against ESMF reference weights. The test suite must:

1. Compare sparse weight matrices (entry-wise and row-sum) and round-tripped
   fields for three grid pairs.
2. Cover both 1st-order (xESMF or ESMPy) and 2nd-order (ESMPy only —
   `CONSERVE_2ND` is not exposed by xESMF).
3. Skip gracefully when Python / esmpy is unavailable.

## Approach

Compute reference weights **live at test time** by calling Python through
`PythonCall` + `CondaPkg`. The test conda environment already exists at
`test/.CondaPkg/pixi.toml` and already pulls in `esmpy` and `xesmf` as
transitive deps of XESMF.jl, so this is essentially free. For 1st-order we
keep using XESMF.jl (simple high-level API). For 2nd-order we drop down to
the raw ESMPy bindings via `pyimport("esmpy")`.

We do **not** ship pre-computed reference `.nc` files. Reasons in §4.

## Recommended path (TL;DR)

Use `PythonCall` + `CondaPkg` (already a transitive test dep through XESMF.jl)
to call `xesmf.Regridder(method="conservative")` for 1st-order and raw
`esmpy.Regrid(..., regrid_method=esmpy.RegridMethod.CONSERVE_2ND, filename=…)`
for 2nd-order, reading weights live at test time. Skip the entire test set
with `@test_skip` if `pyimport_e` fails. Compare weight matrices with rtols
of 5e-12 (1st-order) and 5e-9 (2nd-order). Three grid pairs: global lon-lat,
regional lon-lat, cubed-sphere ne=4 → lon-lat.

---

## 1. Which ESMF interface

Findings on the three candidates:

- **xESMF.** Exposed methods: `bilinear`, `conservative`,
  `conservative_normed`, `patch`, `nearest_s2d`, `nearest_d2s`. **No
  2nd-order.** Confirmed in `xesmf/frontend.py`. `conservative_normed` is
  1st-order with `NormType.FRACAREA` (mask handling), still 1st-order.
- **ESMPy.** Wraps ESMF directly. Supports `RegridMethod.CONSERVE` and
  `RegridMethod.CONSERVE_2ND`. Verbose but fully featured. Already
  installed in this repo's test env
  (`test/.CondaPkg/.pixi/envs/default/lib/python3.14/site-packages/esmpy`).
- **Pre-generated weights files.** Hermetic but vulnerable to silent drift
  as ESMF's intersection algorithm changes between conda-forge bumps. Live
  computation always tests against the version actually installed.

**Recommendation**: live computation through PythonCall, splitting on method:

| Method     | Tool used        | Why                                            |
|------------|------------------|------------------------------------------------|
| 1st-order  | XESMF.jl         | Already wired, high-level, ergonomic.          |
| 2nd-order  | raw ESMPy        | Only path that exposes `CONSERVE_2ND`.         |

For 2nd-order we materialise weights through ESMPy's `filename=` argument
(writes the standard ESMF SCRIP-format netCDF: `S`, `row`, `col`, `frac_a`,
`frac_b`, `area_a`, `area_b`) and then read it back with NCDatasets. This is
the same format `ConservativeRegridding.save_esmf_weights` already produces,
so we get bidirectional comparison for free.

## 2. Julia–Python bridge

XESMF.jl already uses **PythonCall + CondaPkg** (verified by reading its
`__init__()` which calls `pyimport("xesmf")`). The existing test env
(`test/.CondaPkg/pixi.toml`) already declares `xesmf`, `esmpy`, `esmf`, and
`python`. So:

- `test/Project.toml` — add `PythonCall = "6099a3de-…"` (UUID:
  `6099a3de-0909-46bc-a71f-94f0b0c5b3a8`) and `CondaPkg =
  "992eb4ea-…"` (UUID: `992eb4ea-22a4-4c89-a5bb-47a3300528ab`). Both will
  resolve against the conda env that's already there.
- No `CondaPkg.toml` change needed — `esmpy` is already in
  `test/.CondaPkg/pixi.toml`. (If it ever drifts, add it explicitly under
  `[deps]` with `CondaPkg.add("esmpy", channel="conda-forge")`.)
- Use `PythonCall.pyimport` (and `PythonCall.pyimport_e` to detect
  unavailability without raising).

## 3. Test file structure

### Skip-gracefully guard

At the top of the test file, attempt the imports inside `try`/`catch`. If
either `xesmf` or `esmpy` is missing (CI without conda, broken libnetcdf,
etc.), mark every test in the file as `@test_skip` and `@info` the reason.
Concretely:

```julia
const HAS_PYTHON = try
    using PythonCall
    pyimport("esmpy"); pyimport("xesmf"); pyimport("numpy")
    true
catch err
    @warn "Python ESMF stack unavailable; skipping xESMF comparison" err
    false
end
```

All `@testset` bodies wrap their assertions in `if HAS_PYTHON … else
@test_skip …`. This matches the existing style used by other extension
tests in the repo.

### Three grid pairs (each tested at 1st and 2nd order)

(a) **Global lon-lat → lon-lat**, both wrapping at 360° with implicit pole
    fold: `36×18` source → `24×12` destination. Use the same
    `LatitudeLongitudeGrid` constructors as the current test for
    consistency, but also provide a pure-Julia path that builds vertex
    matrices directly so the test does not require Oceananigans.

(b) **Regional lon-lat → lon-lat**, no wrap, no fold:
    longitude `(-30, 30)` × latitude `(-15, 15)`, source `30×15`,
    destination `20×10`. Stresses partial-coverage handling.

(c) **Cubed-sphere ne=4 → lon-lat 36×18.** Builds the cubed-sphere grid via
    ClimaCore (already a test dep) and feeds the per-element corner
    coordinates as an unstructured `esmpy.Mesh` to ESMPy. This is the
    geometry-conversion-heavy case. **Important**: `esmpy.RegridMethod.CONSERVE_2ND`
    does work for Mesh sources/destinations as long as element coordinates are
    on cell centers, but the polemethod must remain `NONE` (the default) and
    we must build the Mesh with `meshloc=esmpy.MeshLoc.ELEMENT`. Verified
    against ESMF 8.9.1 docs.

### Per-grid-pair / per-method assertions

For each `(grid_pair, method) ∈ pairs × {first_order, second_order}`:

1. Build the Julia regridder:

   ```julia
   r = method == :first_order ?
       Regridder(dst, src; normalize=false) :
       Regridder(dst, src; normalize=false, algorithm=SecondOrderConservative())
   ```

   (The `algorithm` kwarg comes from the not-yet-implemented 2nd-order API.
   The plan in `docs/plans/2026-04-26-second-order-…` is the implementing
   PR's responsibility; this test plan only specifies the interface it
   consumes.)

2. Build the ESMF/xESMF reference. For 1st-order, call `xesmf.Regridder` and
   read `.weights.data` (a `scipy.sparse.coo_matrix`) directly. For
   2nd-order, call `esmpy.Regrid(... filename=tmpfile)` and read the netCDF.

3. **Sparse-matrix comparison**: extract `(I, J, V)` from both, sort by
   `(I, J)`, compare:
   - `nnz` agreement: tolerate up to `0.5%` difference in number of
     non-zeros (ESMF can include or exclude tiny-area intersections at
     numerical-zero thresholds; we set our threshold differently).
   - For shared `(I, J)` pairs, `max(abs(V_julia - V_esmf)) < rtol *
     max(abs(V_esmf))` with rtol from the table below.
   - Row sums agree to `1e-13` for cells fully covered by source (most
     interior cells); `frac_b ≈ 1` should match `frac_b ≈ 1` from ESMF.

4. **Round-trip comparison**: regrid two test fields and compare:
   - **Constant field** `f ≡ 1`. Both methods should reproduce 1 to
     `1e-13` for fully-covered destination cells. (For partial-coverage
     cells at the regional boundary we test against `frac_b`, not 1.)
   - **Smooth field** `f(λ, φ) = cos(2λ) sin(φ)` (lon-lat) or `f(x, y, z) =
     x` (cubed-sphere). 1st-order should agree to `1e-12`, 2nd-order to
     `1e-9` (looser, see below).

### Tolerance table

| Grid pair                | Method     | Matrix rtol | Field rtol |
|--------------------------|------------|-------------|------------|
| lon-lat global           | 1st-order  | 5e-12       | 1e-12      |
| lon-lat global           | 2nd-order  | 5e-9        | 5e-9       |
| lon-lat regional         | 1st-order  | 5e-12       | 1e-12      |
| lon-lat regional         | 2nd-order  | 5e-9        | 5e-9       |
| cubed-sphere → lon-lat   | 1st-order  | 1e-10       | 1e-10      |
| cubed-sphere → lon-lat   | 2nd-order  | 5e-8        | 5e-8       |

The cubed-sphere column is looser because face-edge intersections go through
distinct algorithmic paths in ESMF (MOAB) vs CR (great-circle Sutherland-
Hodgman); the discrepancies show up at the ~1e-11 level for 1st-order and
amplify under the gradient stencil for 2nd-order. These numbers come from
similar comparisons in the literature (Ullrich & Taylor 2015, Kritsikis et
al. 2017).

## 4. Where the reference weights live

**Compute live, do not commit.** `esmpy` is already in
`test/.CondaPkg/pixi.toml`, ESMF is not version-stable to 1e-15 so a pinned
reference would silently drift, and the repo's existing
`save_esmf_weights` (NCDatasetsExt) gives us a debug path on disk if we
need it. The skip-gracefully guard handles Python-less CI.

## 5. Concrete code skeleton

```julia
# test/xesmf_comparison.jl
using Test
using ConservativeRegridding
using ConservativeRegridding: Regridder, regrid!
using SparseArrays
using LinearAlgebra
using NCDatasets
import GeometryOps as GO, GeoInterface as GI

# ---------------------------------------------------------------------------
# Skip-gracefully Python guard
# ---------------------------------------------------------------------------
const HAS_PYTHON = try
    @eval using PythonCall
    pyimport("esmpy")
    pyimport("xesmf")
    pyimport("numpy")
    true
catch err
    @warn "Python ESMF stack unavailable; skipping xESMF comparison." err
    false
end

# Lazy aliases — only valid when HAS_PYTHON
function _py()
    @assert HAS_PYTHON
    return (esmpy = pyimport("esmpy"),
            xesmf = pyimport("xesmf"),
            np    = pyimport("numpy"),
            xr    = pyimport("xarray"))
end

# ---------------------------------------------------------------------------
# Grid builders — pure Julia, return (vertex_matrix, polygon_matrix)
# ---------------------------------------------------------------------------
function lonlat_grid(lon_range, lat_range, nlon, nlat; manifold=GO.Spherical())
    lons = collect(range(lon_range...; length = nlon + 1))
    lats = collect(range(lat_range...; length = nlat + 1))
    # corners as (nlon+1, nlat+1) matrix of (lon, lat) tuples
    corners = [(lons[i], lats[j]) for i in 1:nlon+1, j in 1:nlat+1]
    return (lons = lons, lats = lats, corners = corners,
            nlon = nlon, nlat = nlat, manifold = manifold)
end

# Convert a Julia lon-lat grid to a Python xesmf-friendly xarray Dataset
function to_xesmf_grid(g)
    py = _py()
    # Cell centers
    lon_c = 0.5 .* (g.lons[1:end-1] .+ g.lons[2:end])
    lat_c = 0.5 .* (g.lats[1:end-1] .+ g.lats[2:end])
    return py.xr.Dataset(Dict(
        "lon"   => py.xr.DataArray(lon_c; dims = ("x",)),
        "lat"   => py.xr.DataArray(lat_c; dims = ("y",)),
        "lon_b" => py.xr.DataArray(g.lons; dims = ("x_b",)),
        "lat_b" => py.xr.DataArray(g.lats; dims = ("y_b",)),
    ))
end

# ---------------------------------------------------------------------------
# Read xesmf 1st-order weights into Julia (I, J, V)
# ---------------------------------------------------------------------------
function xesmf_weights_first_order(src_g, dst_g; periodic)
    py = _py()
    Rpy = py.xesmf.Regridder(to_xesmf_grid(src_g), to_xesmf_grid(dst_g),
                             "conservative"; periodic = periodic)
    coo = Rpy.weights.data           # scipy.sparse.coo_matrix
    I = pyconvert(Vector{Int}, coo.row) .+ 1   # Python 0-based → Julia 1-based
    J = pyconvert(Vector{Int}, coo.col) .+ 1
    V = pyconvert(Vector{Float64}, coo.data)
    return sparse(I, J, V, dst_g.nlon * dst_g.nlat, src_g.nlon * src_g.nlat)
end

# ---------------------------------------------------------------------------
# Compute esmpy 2nd-order weights, write to tmp .nc, read back
# ---------------------------------------------------------------------------
function esmpy_weights_second_order(src_g, dst_g)
    py = _py()
    src_grid = _build_esmpy_grid(src_g)
    dst_grid = _build_esmpy_grid(dst_g)
    src_field = py.esmpy.Field(src_grid; staggerloc = py.esmpy.StaggerLoc.CENTER)
    dst_field = py.esmpy.Field(dst_grid; staggerloc = py.esmpy.StaggerLoc.CENTER)
    tmp = tempname() * ".nc"
    py.esmpy.Regrid(src_field, dst_field;
                    regrid_method = py.esmpy.RegridMethod.CONSERVE_2ND,
                    filename      = tmp,
                    unmapped_action = py.esmpy.UnmappedAction.IGNORE)
    return _read_esmf_weights_nc(tmp, dst_g.nlon * dst_g.nlat,
                                       src_g.nlon * src_g.nlat), tmp
end

function _read_esmf_weights_nc(path, n_b, n_a)
    NCDataset(path) do ds
        S   = Array(ds["S"][:])
        row = Array(ds["row"][:])           # ESMF row/col are 1-based already
        col = Array(ds["col"][:])
        sparse(row, col, S, n_b, n_a)
    end
end

# ---------------------------------------------------------------------------
# Julia regridder weight matrix in same convention (S = A / dst_areas)
# ---------------------------------------------------------------------------
function julia_weight_matrix(r::Regridder)
    A = r.intersections
    da = r.dst_areas
    Is, Js, Vs = findnz(A)
    return sparse(Is, Js, Vs ./ da[Is], size(A)...)
end

# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------
function compare_sparse(W_jl, W_py; matrix_rtol)
    @test size(W_jl) == size(W_py)
    nnz_jl, nnz_py = nnz(W_jl), nnz(W_py)
    @test abs(nnz_jl - nnz_py) / max(nnz_jl, nnz_py) < 0.005
    diff = W_jl - W_py
    scale = max(maximum(abs, W_py), 1e-300)
    max_abs = maximum(abs, diff.nzval; init = 0.0)
    @info "matrix diff" nnz_jl nnz_py max_abs rel = max_abs / scale
    @test max_abs / scale < matrix_rtol
end

# ---------------------------------------------------------------------------
# Test sets
# ---------------------------------------------------------------------------
@testset "xESMF comparison" begin
    if !HAS_PYTHON
        @test_skip "Python ESMF unavailable"
        return
    end

    grid_pairs = [
        (name = "global lon-lat",
         src  = lonlat_grid((0, 360), (-90, 90), 36, 18),
         dst  = lonlat_grid((0, 360), (-90, 90), 24, 12),
         periodic = true),
        (name = "regional lon-lat",
         src  = lonlat_grid((-30, 30), (-15, 15), 30, 15),
         dst  = lonlat_grid((-30, 30), (-15, 15), 20, 10),
         periodic = false),
        # Cubed-sphere case factored into a separate test set so we can use
        # ClimaCore + esmpy.Mesh
    ]

    for gp in grid_pairs
        @testset "$(gp.name)" begin
            src_polys = lonlat_polygons(gp.src)
            dst_polys = lonlat_polygons(gp.dst)

            @testset "1st-order" begin
                r_jl = Regridder(dst_polys, src_polys; normalize = false)
                W_jl = julia_weight_matrix(r_jl)
                W_py = xesmf_weights_first_order(gp.src, gp.dst;
                                                 periodic = gp.periodic)
                compare_sparse(W_jl, W_py; matrix_rtol = 5e-12)

                # Field round-trip
                src_field = smooth_field(gp.src)
                dst_jl = similar(src_field, length(r_jl.dst_areas)); fill!(dst_jl, 0)
                regrid!(dst_jl, r_jl, src_field)
                dst_py = pyconvert(Vector{Float64}, W_py * src_field) # via SparseArrays
                @test maximum(abs, dst_jl .- dst_py) /
                      max(maximum(abs, dst_py), 1e-300) < 1e-12
            end

            @testset "2nd-order" begin
                # NB. depends on Regridder(...; algorithm = SecondOrderConservative())
                r_jl = Regridder(dst_polys, src_polys;
                                 normalize = false,
                                 algorithm = ConservativeRegridding.SecondOrderConservative())
                W_jl = julia_weight_matrix(r_jl)
                W_py, tmp = esmpy_weights_second_order(gp.src, gp.dst)
                try
                    compare_sparse(W_jl, W_py; matrix_rtol = 5e-9)
                    src_field = smooth_field(gp.src)
                    dst_jl = zeros(length(r_jl.dst_areas))
                    regrid!(dst_jl, r_jl, src_field)
                    dst_py = W_py * src_field
                    @test maximum(abs, dst_jl .- dst_py) /
                          max(maximum(abs, dst_py), 1e-300) < 5e-9
                finally
                    rm(tmp; force = true)
                end
            end
        end
    end

    @testset "cubed-sphere ne=4 → lon-lat 36×18" begin
        # Build CR side via ClimaCore, ESMF side via esmpy.Mesh from
        # ClimaCore's element corner coordinates.  Tolerances 1e-10 / 5e-8.
        # Implementation detail in the followup PR — sketch in section 6.
    end
end
```

Helpers `lonlat_polygons`, `smooth_field`, `_build_esmpy_grid` are
~30 lines each and follow the patterns in `test/usecases/simple.jl` and
`ext/ConservativeRegriddingNCDatasetsExt.jl`. Total file is ~250 lines.

## 6. Things that will surprise the implementer

- **xESMF only does 1st-order.** Confirmed in `xesmf/frontend.py` and the
  upstream issue tracker. Do not waste time looking for a `method=` string
  that triggers 2nd-order.
- **ESMPy index convention.** The `row` and `col` in ESMF weight files are
  **1-based**, but xESMF's in-memory `Regridder.weights.data` (a
  `scipy.sparse.coo_matrix`) is **0-based**. Off-by-one bugs will eat a
  morning.
- **xESMF auto-orientation.** `xesmf.Regridder(periodic=True)` fills the
  global pole with synthetic cells; we have to set `periodic=False` for
  regional grids. CR has no equivalent flag — get this wrong and the global
  test "fails" because xESMF added 2 extra rows.
- **CONSERVE_2ND silently disables polemethod.** ESMF's
  `polemethod=esmpy.PoleMethod.NONE` is forced for `CONSERVE_2ND`. Don't
  set anything else or `Regrid` will raise. (Default for Grid construction
  is already `NONE`, so this is mostly a "leave it alone" warning.)
- **CONSERVE_2ND requires a gradient stencil.** ESMF computes that from the
  source grid's **cell centroids**. CR's centroid definition for the 2nd-order
  API is "vertex-mean" by default. ESMF uses an analytical formula on the
  unit sphere that is **not** vertex-mean. This is the dominant source of
  the ~1e-9 disagreement and motivates the looser 2nd-order rtol.
- **`unmapped_action`.** Set
  `unmapped_action=esmpy.UnmappedAction.IGNORE`. The default raises if any
  destination cell is unreachable — happens for partial regional grids and
  for any 2nd-order destination cell at a corner where the gradient stencil
  is degenerate. We compare only on the cells both methods covered.
- **Cubed sphere as `esmpy.Mesh`.** ESMF distinguishes `Grid` (logically
  rectangular) from `Mesh` (unstructured). Cubed-sphere cells must go in a
  Mesh, with element connectivity flattened into `(node_count, element_count,
  element_conn)` arrays. `meshloc=esmpy.MeshLoc.ELEMENT` and
  `staggerloc` is irrelevant (Mesh always uses element centers). Test grid
  ne=4 has 6 × 16 = 96 cells; the connectivity array is small.
- **MPI is not required.** ESMPy works in serial; ESMF detects this when
  `MPI_Init` has not been called. No `mpiexec` wrapper needed.
- **`ESMFMKFILE` env var.** Setting this is normally only needed when
  building ESMPy from source. The conda-forge `esmpy` package sets it via
  its activation script; CondaPkg-managed envs may not run that script,
  so if `pyimport("esmpy")` errors with "ESMFMKFILE not set", set it
  manually:
  ```julia
  ENV["ESMFMKFILE"] = joinpath(CondaPkg.envdir(), "lib", "esmf.mk")
  ```
  Add this guard to the `try` block in §3 — it's cheap and prevents one
  whole class of CI failures.
- **Existing `test/usecases/xesmf_comparison.jl`** uses `XESMF.jl` (the
  high-level wrapper) and Oceananigans grids. The new file at
  `test/xesmf_comparison.jl` should replace it. Update `runtests.jl`:
  remove the `usecases/xesmf_comparison.jl` line, add a top-level
  `@safetestset "Comparison: XESMF/ESMF" begin include("xesmf_comparison.jl") end`.
- **Determinism across ESMF versions.** Pin esmpy in the test env once we're
  comfortable with the tolerances:
  `CondaPkg.add("esmpy"; version="=8.9.1", channel="conda-forge")`.
  Do this in a follow-up after the first green CI run to avoid re-running
  the comparison every time conda-forge rolls a patch.
- **Float type.** ESMPy weights are `float64` in the .nc file. CR uses
  Float64 by default. If the second-order API ends up parameterised on
  `T<:AbstractFloat`, force `T = Float64` in this test and add a separate
  type-stability test elsewhere.
- **Cubed-sphere weight ordering.** Both CR and ESMPy index cubed-sphere
  cells by `(face-1)*ne² + j*ne + i`-style flattening, but ESMPy's flatten
  order depends on how we feed the Mesh. Build the Mesh with the **same**
  iteration order CR uses internally
  (`IndexOffsetQuadtreeCursor` scheme) — otherwise rows and columns are
  permuted and the comparison appears to fail. Sanity-check via the
  constant-field test before trusting matrix-entry comparisons.
