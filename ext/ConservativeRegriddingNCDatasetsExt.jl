"""
NCDatasets extension for ConservativeRegridding.jl — ESMF offline-weights export.

Activated when `using NCDatasets` is called alongside ConservativeRegridding.
Provides `save_esmf_weights(path, regridder; ...)` which writes the standard
ESMF weight-file format (S, row, col, frac_a, frac_b, area_a, area_b).
"""
module ConservativeRegriddingNCDatasetsExt

using ConservativeRegridding
using ConservativeRegridding: Regridder
using NCDatasets
using SparseArrays: rowvals, nonzeros, nzrange
using Dates: now

function ConservativeRegridding.save_esmf_weights(
        path::AbstractString, r::Regridder;
        src_grid_name::AbstractString = "source",
        dst_grid_name::AbstractString = "destination",
        src_shape::Union{Nothing, Tuple} = nothing,
        dst_shape::Union{Nothing, Tuple} = nothing,
    )
    A         = r.intersections
    src_areas = r.src_areas
    dst_areas = r.dst_areas

    n_a = length(src_areas)
    n_b = length(dst_areas)
    size(A) == (n_b, n_a) ||
        error("intersections shape $(size(A)) != (n_b=$n_b, n_a=$n_a)")

    rows = rowvals(A)
    vals = nonzeros(A)
    n_s  = length(vals)

    row_out = Vector{Int64}(undef, n_s)
    col_out = Vector{Int64}(undef, n_s)
    S_out   = Vector{Float64}(undef, n_s)
    covered_src = zeros(Float64, n_a)
    covered_dst = zeros(Float64, n_b)

    idx = 0
    for col in 1:n_a
        for ptr in nzrange(A, col)
            idx += 1
            d = rows[ptr]
            a = Float64(vals[ptr])
            row_out[idx] = d
            col_out[idx] = col
            S_out[idx]   = a / Float64(dst_areas[d])
            covered_src[col] += a
            covered_dst[d]   += a
        end
    end

    frac_a = covered_src ./ Float64.(src_areas)
    frac_b = covered_dst ./ Float64.(dst_areas)

    mkpath(dirname(path))
    NCDataset(path, "c") do ds
        ds.dim["n_s"] = n_s
        ds.dim["n_a"] = n_a
        ds.dim["n_b"] = n_b

        defVar(ds, "S",      S_out,                ("n_s",); attrib = ["long_name" => "weight value"])
        defVar(ds, "row",    row_out,              ("n_s",); attrib = ["long_name" => "destination cell index (1-based)"])
        defVar(ds, "col",    col_out,              ("n_s",); attrib = ["long_name" => "source cell index (1-based)"])
        defVar(ds, "frac_a", frac_a,               ("n_a",); attrib = ["long_name" => "source cell fraction covered"])
        defVar(ds, "frac_b", frac_b,               ("n_b",); attrib = ["long_name" => "destination cell fraction covered"])
        defVar(ds, "area_a", Float64.(src_areas),  ("n_a",); attrib = ["long_name" => "source cell areas"])
        defVar(ds, "area_b", Float64.(dst_areas),  ("n_b",); attrib = ["long_name" => "destination cell areas"])

        ds.attrib["title"]            = "ConservativeRegridding.jl weights (ESMF format)"
        ds.attrib["created_by"]       = "ConservativeRegridding.save_esmf_weights"
        ds.attrib["created_at"]       = string(now())
        ds.attrib["source_grid"]      = String(src_grid_name)
        ds.attrib["destination_grid"] = String(dst_grid_name)
        ds.attrib["normalization"]    = "destarea"
        ds.attrib["map_method"]       = "Conservative remapping"
        if src_shape !== nothing
            ds.attrib["source_grid_shape"] = collect(Int64.(src_shape))
        end
        if dst_shape !== nothing
            ds.attrib["destination_grid_shape"] = collect(Int64.(dst_shape))
        end
    end
    return path
end

end # module
