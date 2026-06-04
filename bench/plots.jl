# CairoMakie rendering of benchmark results. Kept separate from the benchmark drivers so the
# drivers (scaling.jl, xesmf.jl) stay free of plotting/conda dependencies. Operates purely on
# the primitive NamedTuple rows produced by the drivers — never loads ConservativeRegridding —
# so it can plot PR and base results side by side without a package-version clash.

using CairoMakie
using Serialization
using Printf

const _FAMILY_COLOR = Dict("Oceananigans" => :steelblue, "Healpix" => :darkorange)
const _METHOD_COLOR = Dict("CR" => :steelblue, "CR-serial" => :seagreen, "XESMF" => :firebrick)

_methodlabel(m) = m == "CR" ? "ConservativeRegridding (threaded)" :
                  m == "CR-serial" ? "ConservativeRegridding (serial)" : "XESMF / ESMF"

_fmt_time(t) = t >= 1 ? @sprintf("%.2f s", t) :
               t >= 1e-3 ? @sprintf("%.1f ms", t * 1e3) : @sprintf("%.0f µs", t * 1e6)
_humancells(n) = n >= 1000 ? @sprintf("%.0fk", n / 1000) : string(n)

# --- Part 1: construction scaling, one line per grid family ---------------------------------
function plot_scaling(rows; path)
    rows = filter(r -> r.method == "CR", rows)
    fig = Figure(size = (820, 540))
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10,
        xlabel = "destination cells (source at half-resolution)",
        ylabel = "construction time (s)",
        title = "Regridder construction scaling by grid family")
    for fam in sort(unique(r.family for r in rows))
        sub = sort(filter(r -> r.family == fam, rows); by = r -> r.ncells_dst)
        isempty(sub) && continue
        scatterlines!(ax, [Float64(r.ncells_dst) for r in sub], [r.time_s for r in sub];
            label = fam, color = get(_FAMILY_COLOR, fam, :black), markersize = 11, linewidth = 2)
    end
    !isempty(rows) && axislegend(ax; position = :lt)
    nthreads = isempty(rows) ? 0 : rows[1].nthreads
    Label(fig[2, 1], "Julia threads: $nthreads · lower is better";
        fontsize = 11, color = :gray40, tellwidth = false)
    save(path, fig)
    return path
end

# --- Part 2: CR (threaded & serial) vs XESMF construction time ------------------------------
function plot_xesmf(rows; path)
    fig = Figure(size = (840, 540))
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10,
        xlabel = "destination cells (source at half-resolution)",
        ylabel = "construction time (s)",
        title = "Construction time: ConservativeRegridding vs XESMF (Oceananigans)")
    for m in ["CR", "CR-serial", "XESMF"]
        sub = sort(filter(r -> r.method == m, rows); by = r -> r.ncells_dst)
        isempty(sub) && continue
        scatterlines!(ax, [Float64(r.ncells_dst) for r in sub], [r.time_s for r in sub];
            label = _methodlabel(m), color = _METHOD_COLOR[m], markersize = 11, linewidth = 2)
    end
    axislegend(ax; position = :lt)
    nthreads = isempty(rows) ? 0 : rows[1].nthreads
    Label(fig[2, 1], "CR threaded uses $nthreads Julia threads; XESMF weight-gen is ESMF-controlled · lower is better";
        fontsize = 11, color = :gray40, tellwidth = false)
    save(path, fig)
    return path
end

# --- Part 4 (CI): PR-vs-base construction-time ratio, one line per family -------------------
function plot_pr_vs_master(pr_rows, base_rows; path, base_label = "base")
    pr = filter(r -> r.method == "CR", pr_rows)
    basemap = Dict((r.family, r.tier) => r.time_s for r in base_rows if r.method == "CR")
    fig = Figure(size = (860, 540))
    ax = Axis(fig[1, 1]; xscale = log10,
        xlabel = "destination cells (source at half-resolution)",
        ylabel = "construction time ratio  (PR / $base_label)",
        title = "Construction time change: PR vs $base_label")
    hspan!(ax, 1.0, 1e3; color = (:tomato, 0.10))      # slower than base
    hspan!(ax, 1e-3, 1.0; color = (:seagreen, 0.10))   # faster than base
    hlines!(ax, [1.0]; color = :gray50, linestyle = :dash)
    ratios = Float64[]
    plotted = false
    for fam in sort(unique(r.family for r in pr))
        sub = sort(filter(r -> r.family == fam, pr); by = r -> r.ncells_dst)
        xs = Float64[]; ys = Float64[]
        for r in sub
            bt = get(basemap, (r.family, r.tier), NaN)
            isfinite(bt) && bt > 0 || continue
            push!(xs, r.ncells_dst); push!(ys, r.time_s / bt); push!(ratios, r.time_s / bt)
        end
        isempty(xs) && continue
        scatterlines!(ax, xs, ys; label = fam, color = get(_FAMILY_COLOR, fam, :black),
            markersize = 11, linewidth = 2)
        plotted = true
    end
    if plotted
        axislegend(ax; position = :rt)
        hi = max(1.15, maximum(ratios) * 1.1)
        lo = min(0.85, minimum(ratios) * 0.9)
        ylims!(ax, lo, hi)
        Label(fig[2, 1], "below 1.0 (green) = PR faster · above 1.0 (red) = PR slower · shared CI runners are noisy";
            fontsize = 11, color = :gray40, tellwidth = false)
    else
        text!(ax, 0.5, 0.5; text = "no base data\n(base build failed or unavailable)",
            space = :relative, align = (:center, :center), color = :gray50)
        ylims!(ax, 0.0, 2.0)
    end
    save(path, fig)
    return path
end

# --- Markdown summaries (for the PR comment) ------------------------------------------------
function summary_markdown(pr_rows, base_rows = NamedTuple[]; base_label = "base")
    pr = filter(r -> r.method == "CR", pr_rows)
    base = Dict((r.family, r.tier) => r for r in base_rows if r.method == "CR")
    io = IOBuffer()
    println(io, "| family | dst cells | PR | $base_label | ratio | PR allocs | nnz |")
    println(io, "|---|--:|--:|--:|--:|--:|--:|")
    for r in sort(pr; by = r -> (r.family, r.ncells_dst))
        b = get(base, (r.family, r.tier), nothing)
        bt = isnothing(b) ? "—" : _fmt_time(b.time_s)
        ratio = isnothing(b) ? "—" : @sprintf("%.2f×", r.time_s / b.time_s)
        println(io, "| $(r.family) | $(r.ncells_dst) | $(_fmt_time(r.time_s)) | $bt | $ratio | $(r.allocs) | $(r.nnz) |")
    end
    return String(take!(io))
end

function xesmf_summary_markdown(rows)
    tiers = sort(unique((r.tier, r.ncells_dst) for r in rows); by = t -> t[2])
    bym = Dict((r.method, r.tier) => r for r in rows)
    io = IOBuffer()
    println(io, "| dst cells | CR (threaded) | CR (serial) | XESMF | CR/XESMF |")
    println(io, "|--:|--:|--:|--:|--:|")
    for (tier, ncells) in tiers
        cr = get(bym, ("CR", tier), nothing)
        crs = get(bym, ("CR-serial", tier), nothing)
        xe = get(bym, ("XESMF", tier), nothing)
        f(x) = isnothing(x) ? "—" : _fmt_time(x.time_s)
        speed = (isnothing(cr) || isnothing(xe)) ? "—" : @sprintf("%.2f×", cr.time_s / xe.time_s)
        println(io, "| $ncells | $(f(cr)) | $(f(crs)) | $(f(xe)) | $speed |")
    end
    return String(take!(io))
end
