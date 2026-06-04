# Benchmarks

Regridder-**construction**-time benchmarks for ConservativeRegridding.jl. Everything here uses
the `bench/` environment (`Project.toml` + `CondaPkg.toml`), a member of the repo workspace.

## What's measured

| File | What | Grids | Needs conda? |
|---|---|---|---|
| `scaling.jl` | construction time vs grid **size** | Oceananigans LatLon, Healpix | no |
| `xesmf.jl` | ConservativeRegridding vs **XESMF** (ESMF) construction | Oceananigans only¹ | yes (esmpy/ESMF) |
| `compare.jl` | **PR-vs-base** construction scaling (two-ref) | Oceananigans, Healpix | no |
| `spectral_element.jl` | SE↔FV construction vs FV→FV baseline | ClimaCore + Oceananigans | no |

¹ XESMF only regrids Oceananigans grids (via its extension), hence the Oceananigans-only scope.

`grids.jl` holds the grid-pair factories (coarse source → finer destination, source at half the
destination resolution) shared by all of the above. Tiers sweep size: Oceananigans `Ny ∈
{16,32,64,128,256}` (cells = 2·Ny²) and Healpix `nside ∈ {8,16,32,64,128}` (cells = 12·nside²).

## Run locally

```bash
# Full suite (scaling + CR-vs-XESMF) → graphs + tables in bench/results/
julia --project=bench bench/run.jl            # all tiers
julia --project=bench bench/run.jl 50000      # cap destination cells per tier

# PR-vs-base comparison against the current checkout (writes bench/results/pr_vs_master.png)
JULIA_NUM_THREADS=4 julia --project=bench bench/compare.jl main bench/results 50000
```

Outputs (git-ignored) land in `bench/results/`: `scaling.png`, `xesmf.png`, `pr_vs_master.png`,
plus `*.jls` (serialized primitive rows) and `summary.md`.

## CI (`.github/workflows/benchmark.yml`)

Modeled on Makie's PR-vs-base benchmark job. **Label-gated**: add the **`run benchmarks`** label
to a PR (or use *Run workflow* / `workflow_dispatch`) to trigger it. The job

1. runs `compare.jl` to benchmark the **PR** and the **base** ref — the driver always comes from
   the PR checkout; only the ConservativeRegridding package is swapped (base via a detached git
   worktree, each ref benchmarked in an isolated subprocess so the loaded package is provable);
2. runs the CR-vs-XESMF comparison on the PR ref (conda; isolated with `continue-on-error`);
3. pushes the graphs to a **separate** assets repo — `JuliaGeo/JuliaGeoBenchmarkResults`, branch
   **`ConservativeRegridding`**, under `pr-<N>/` — and embeds them inline in a single sticky PR
   comment. Keeping the PNG blobs out of this repo avoids bloating its git history.

The cross-repo push uses an **SSH deploy key** (write-scoped to the assets repo only) stored as the
`BENCHMARK_ASSETS_DEPLOY_KEY` secret — the stock `GITHUB_TOKEN` can't push to another repo. The
sticky comment is still posted to this repo with the stock token. The full-resolution graphs are
also uploaded as a workflow **artifact**. Fork PRs get neither the secret nor a writable token, so
the push/comment is skipped there (the artifact still uploads).
