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
using SparseArrays: findnz

function ConservativeRegridding.save_esmf_weights(
        path::AbstractString, r::Regridder;
        src_grid_name::AbstractString = "source",
        dst_grid_name::AbstractString = "destination",
        src_shape::Union{Nothing, Tuple} = nothing,
        dst_shape::Union{Nothing, Tuple} = nothing,
        created_at::Union{Nothing, AbstractString} = nothing,
    )
    # Materialize to CPU arrays: NCDatasets expects host memory, and A /
    # src_areas / dst_areas may live on a GPU. `Array(::Array)` is a no-op.
    A         = r.intersections
    src_areas = Float64.(Array(r.src_areas))
    dst_areas = Float64.(Array(r.dst_areas))

    row, col, vals = map(Array, findnz(A))
    S      = Float64.(vals) ./ dst_areas[row]
    frac_a = Array(vec(sum(A; dims = 1))) ./ src_areas
    frac_b = Array(vec(sum(A; dims = 2))) ./ dst_areas

    mkpath(dirname(path))
    NCDataset(path, "c") do ds
        ds.dim["n_s"] = length(vals)
        ds.dim["n_a"] = length(src_areas)
        ds.dim["n_b"] = length(dst_areas)

        defVar(ds, "S",      S,          ("n_s",); attrib = ["long_name" => "weight value"])
        defVar(ds, "row",    row,        ("n_s",); attrib = ["long_name" => "destination cell index (1-based)"])
        defVar(ds, "col",    col,        ("n_s",); attrib = ["long_name" => "source cell index (1-based)"])
        defVar(ds, "frac_a", frac_a,     ("n_a",); attrib = ["long_name" => "source cell fraction covered"])
        defVar(ds, "frac_b", frac_b,     ("n_b",); attrib = ["long_name" => "destination cell fraction covered"])
        defVar(ds, "area_a", src_areas,  ("n_a",); attrib = ["long_name" => "source cell areas"])
        defVar(ds, "area_b", dst_areas,  ("n_b",); attrib = ["long_name" => "destination cell areas"])

        ds.attrib["title"]            = "ConservativeRegridding.jl weights (ESMF format)"
        ds.attrib["created_by"]       = "ConservativeRegridding.save_esmf_weights"
        ds.attrib["source_grid"]      = String(src_grid_name)
        ds.attrib["destination_grid"] = String(dst_grid_name)
        ds.attrib["normalization"]    = "destarea"
        ds.attrib["map_method"]       = "Conservative remapping"
        created_at === nothing || (ds.attrib["created_at"]            = String(created_at))
        src_shape  === nothing || (ds.attrib["source_grid_shape"]      = collect(Int64.(src_shape)))
        dst_shape  === nothing || (ds.attrib["destination_grid_shape"] = collect(Int64.(dst_shape)))
    end
    return path
end

end # module
