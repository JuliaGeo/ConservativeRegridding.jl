# Two-ref PR-vs-base regridder-construction comparison — the core of the benchmark CI.
#
#     julia --project=<env-with-CairoMakie> bench/compare.jl <base_ref> <outdir> [maxcells] [base_label]
#
# Strategy (Makie-style: the driver comes from the PR, only the package is swapped per ref):
#   * For each ref, build a fresh CONDA-FREE temp environment that `Pkg.develop`s the
#     ConservativeRegridding source for that ref, then run THIS repo's bench/scaling.jl in an
#     isolated subprocess against it. The driver file is always the PR's; only the loaded
#     ConservativeRegridding differs. scaling.jl prints `pathof(ConservativeRegridding)` so each
#     ref's package is provable in the logs.
#   * The base ref's source comes from a detached git worktree under .worktrees/base. (The base
#     ref need not contain bench/ — the driver is invoked from the PR checkout.)
#   * This parent process only loads CairoMakie + Serialization (via plots.jl) — never CR — so
#     there is no PR-vs-base package-version clash in the plotting step.
#   * Thread count is pinned identically for both refs (JULIA_NUM_THREADS) for a fair compare.

using Serialization
include(joinpath(@__DIR__, "plots.jl"))

const REPO     = normpath(joinpath(@__DIR__, ".."))
const NTHREADS = get(ENV, "JULIA_NUM_THREADS", "4")

base_ref   = length(ARGS) >= 1 ? ARGS[1] : "main"            # git ref/SHA to check out for base
outdir     = length(ARGS) >= 2 ? ARGS[2] : joinpath(@__DIR__, "results")
maxcells   = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 50_000
base_label = length(ARGS) >= 4 ? ARGS[4] : base_ref           # human label for plots/tables (e.g. "main")
mkpath(outdir)

# A conda-free environment that dev-depends on the ConservativeRegridding at `crpath`, plus the
# packages scaling.jl needs (Oceananigans + Healpix activate CR's extensions).
function build_scaling_env(crpath)
    env = mktempdir()
    code = """
    using Pkg
    Pkg.develop(path = raw"$(crpath)")
    Pkg.add(["Oceananigans", "Healpix", "Chairmarks", "GeometryOps"])
    Pkg.instantiate()
    """
    run(`julia --startup-file=no --project=$(env) -e $(code)`)
    return env
end

function run_ref(label, crpath, outjls)
    @info "benchmarking ref" label crpath
    env = build_scaling_env(crpath)
    driver = joinpath(REPO, "bench", "scaling.jl")
    run(`julia --startup-file=no --project=$(env) --threads=$(NTHREADS) $(driver) $(outjls) $(maxcells)`)
    return deserialize(outjls)
end

# PR ref = this checkout.
pr_rows = run_ref("PR", REPO, joinpath(outdir, "pr.jls"))

# Base ref via a detached worktree (.worktrees/ is gitignored). Try the ref as given, then
# fall back to origin/<ref> (CI fetches base as a remote-tracking ref, not a local branch).
function add_base_worktree(base_wt, ref)
    ispath(base_wt) && run(`git -C $(REPO) worktree remove --force $(base_wt)`)
    try
        run(`git -C $(REPO) worktree add --force --detach $(base_wt) $(ref)`)
    catch
        run(`git -C $(REPO) worktree add --force --detach $(base_wt) origin/$(ref)`)
    end
end

base_wt = joinpath(REPO, ".worktrees", "base")
base_rows = try
    add_base_worktree(base_wt, base_ref)
    run_ref("base ($base_label)", base_wt, joinpath(outdir, "base.jls"))
catch e
    @error "base benchmark failed; degrading to PR-only comparison" exception = (e, catch_backtrace())
    NamedTuple[]
finally
    try
        ispath(base_wt) && run(`git -C $(REPO) worktree remove --force $(base_wt)`)
    catch
    end
end

plot_pr_vs_master(pr_rows, base_rows; path = joinpath(outdir, "pr_vs_master.png"), base_label = base_label)
open(io -> println(io, summary_markdown(pr_rows, base_rows; base_label = base_label)),
    joinpath(outdir, "summary.md"), "w")
@info "comparison written" outdir
