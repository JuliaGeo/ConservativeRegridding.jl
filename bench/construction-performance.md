# Regridder construction performance

An investigation into where `Regridder` construction spends its time on spherical grids, and
why ESMF (via XESMF) outruns ConservativeRegridding at conservative weight generation. Prompted
by the CR-vs-XESMF benchmark (`bench/xesmf.jl`), which shows XESMF ~3× faster than threaded CR
and ~7× faster than serial CR across all tiers on the CI runner.

## Method

Each phase of `Regridder(Spherical(), dst, src)` was timed separately (min of several reps) for
the Oceananigans LatLon tiers, threaded (4 Julia threads) and serial, plus a statistical
`Profile` of the dominant phase. Numbers below are on an Apple-Silicon laptop, so absolute ms are
~10× faster than the shared ubuntu CI runner — but the **phase proportions and speedup ratios are
platform-independent**. Reproduce with `bench/profile_cr.jl` (a thin wrapper over the internal
phase functions).

The construction phases are: **treeify** → **candidate search** (dual depth-first search over the
two quadtrees, extent-pruned) → **COO assembly** (per candidate pair: Sutherland–Hodgman clip +
spherical area) → **sparse build** → **per-cell areas**.

## Where the time went (before)

Oceananigans `Ny=128` → 32,768 dst cells, 8,192 src cells; 274,688 candidate pairs, 48,640 nnz:

| phase | threaded | serial | parallel scaling |
| --- | ---: | ---: | ---: |
| candidate search | 74.2 ms (52%) | 277.6 ms (55%) | 3.7× |
| COO assembly | 57.8 ms (40%) | 209.8 ms (42%) | 3.6× |
| per-cell areas | 10.6 ms (7%) | 11.7 ms (2%) | ~1× (serial) |
| treeify + sparse | 1.1 ms (<1%) | 0.9 ms (<1%) | — |
| **total** | **143.8 ms** | **500.0 ms** | |

## Three root causes of the gap

1. **Loose broad phase → 5.6× wasted clips.** The dual DFS emits **274,688** candidate pairs but
   only **48,640** produce nonzero area — ~82% of the (expensive) Sutherland–Hodgman clips return
   empty. Spherical caps over-approximate lat/lon cells (a circle bounding a rectangle, worse near
   the poles), so the broad phase passes far more pairs than truly overlap.

2. **~573 MB of allocation in assembly (~2 KB per clip).** Every clip allocates fresh polygon
   buffers (`grow_to!`, `array_new_memory`, `GenericMemory`). ESMF is allocation-free Fortran.
   This allocator lives in GeometryOps (`_intersection_sutherland_hodgman`), upstream of CR.

3. **Trig-heavy exact spherical math — some of it needless.** The profile was dominated by
   `_spherical_cap` (CR's broad-phase extent), which did **4 `slerp`s + 8 `spherical_distance`s
   per node**, all `atan`/`cos`/`sqrt`. The great-circle midpoint at `t=0.5` is *exactly*
   `normalize(a+b)` (no slerp), and the max angular distance is `acos` of the **min cosine** (dot
   products + one `acos`) — removable trig with no accuracy loss.

### Secondary findings

- The `# this is just serial … big bottleneck` TODO at `intersection_areas.jl:88` is **stale** —
  candidate search parallelizes 3.7×.
- `areas` (per-cell) runs serially (~7% of threaded total) — free parallelism left on the table.
- `getchild` allocates per subdivision (the lazy quadtree rebuilds nodes each traversal).

## Fix applied: trig-free broad-phase extents

Landed in **#112** (against `as/parallelize-sparsematrix`). Rewrote `_spherical_cap` and the
`ExplicitPolygonGrid{<:Spherical}` extent (`src/trees/grids.jl`):

- **Great-circle edge midpoint** `slerp(a, b, 0.5)` → `normalize(a + b)` — provably identical for
  non-antipodal `a, b` (verified to ~15 digits).
- **Cap radius** `max(spherical_distance(center, ·))` → `acos(clamp(min(dot(center, ·)), -1, 1))`
  — one `acos` instead of per-point trig.
- **Antipodal guard** (`_midcos`): on a large index-rectangle, an edge can span ~180° so its two
  corners are antipodal and `normalize(a+b)` is `NaN`; there the endpoints already force the cap
  wide, so the midpoint is dropped from the radius. (GeometryOps' `slerp` returned a finite
  arbitrary point here; the explicit guard is more principled and avoids the `NaN`.)

### Result

Same `Ny=128` workload, after the change:

| phase | threaded | serial |
| --- | ---: | ---: |
| candidate search | **18.5 ms** (21%) | **64.1 ms** (23%) |
| COO assembly | 59.6 ms (66%) | 203.3 ms (73%) |
| per-cell areas | 10.9 ms (12%) | 11.2 ms (4%) |
| **total** | **89.9 ms** | **279.6 ms** |

- **Candidate search: 4.0× faster (threaded), 4.3× (serial).**
- **Total construction: 1.6× faster (threaded), 1.79× (serial).**
- **Correctness preserved:** candidate count (274,688) and nnz (48,640) are byte-identical before
  and after; the `cell_range_extent spherical: bounding cap` characterization test, the dual-DFS
  tests, and the regridding-conservation use-case tests all pass unchanged.

Candidate search drops from ~55% to ~21% of construction — **COO assembly is now the bottleneck**
(66–73%).

## Remaining gap / next levers

Now that assembly dominates, the next targets (in rough leverage order):

1. **Allocation-free clipping (upstream).** Reusing buffers in GeometryOps' Sutherland–Hodgman
   would attack the 573 MB / ~2 KB-per-clip head-on — the single biggest assembly cost.
2. **Tighten the broad phase** to cut the 5.6× false-positive clips (a cheap secondary reject, or
   a bound tighter than the spherical cap before the full clip).
3. **Parallelize `areas`**, and fix the stale candidate-search TODO comment.
