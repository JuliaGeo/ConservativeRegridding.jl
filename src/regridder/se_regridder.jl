"""
    SEtoFVRegridder{W,A,V} <: AbstractRegridder

Regridder from a spectral element (SE) grid to a finite volume (FV) grid.

The `weight_matrix` is a sparse matrix of size `(N_fv, N_se_nodes)` whose entries
are the SE Jacobian integration weights `W[e,i,j]` for each SE node that falls
inside a given FV cell.  Regridding computes `dst = weight_matrix * src / dst_areas`.
"""
struct SEtoFVRegridder{W, A, V} <: AbstractRegridder
    weight_matrix :: W
    dst_areas :: A
    dst_temp :: V
    src_temp :: V
end

"""
    FVtoSERegridder{W,V,N} <: AbstractRegridder

Regridder from a finite volume (FV) grid to a spectral element (SE) grid.

The `weight_matrix` is a sparse matrix of size `(N_se_nodes, N_fv)`.

Two construction paths share this struct:

- **Simplified (`method=:node_in_polygon`)**: entries are 1 where an SE node falls
  inside a FV cell. `inv_node_weights = nothing`. Regridding computes
  `dst = weight_matrix * src` (no normalization). Each SE node receives the value
  of its containing FV cell. Output is automatically continuous because every
  duplicate of a physical node maps to the same cell.

- **Principled (`method=:polygon_intersection`)**: entries are
  `B(k, (e, i, j)) = ∫_{k∩e} ϕᵢ(ξ) ϕⱼ(η) dA_phys` (PDF Eq. 30, with the
  Jacobian cancellation from change of variables). `inv_node_weights[n] = 1/Wᵉᵢⱼ`
  for flat node index `n = (e, i, j)`. Regridding computes
  `dst = (weight_matrix * src) .* inv_node_weights`. Output is NOT automatically
  continuous; downstream consumers should apply DSS (the field-level `regrid!`
  in the ClimaCore extension does this automatically).
"""
struct FVtoSERegridder{W, V, N} <: AbstractRegridder
    weight_matrix :: W
    dst_temp :: V
    src_temp :: V
    "Per-node `1/Wᵉᵢⱼ` (principled), or `nothing` (simplified)"
    inv_node_weights :: N
end

"""
    SEtoSERegridder{W,A,V} <: AbstractRegridder

Regridder from a spectral element (SE) grid to another SE grid.

The `weight_matrix` is a sparse matrix of size `(N_dst_elements, N_src_nodes)` whose
entries are the source SE Jacobian weights.  Regridding first computes per-element
averages `elem_vals = weight_matrix * src / dst_element_areas`, then broadcasts each
element value to all `Nq_dst^2` destination nodes within that element.
"""
struct SEtoSERegridder{W, A, V} <: AbstractRegridder
    weight_matrix :: W
    dst_element_areas :: A
    Nq_dst :: Int
    dst_temp :: V
    src_temp :: V
end

Base.size(r::SEtoFVRegridder, args...) = size(r.weight_matrix, args...)
Base.size(r::FVtoSERegridder, args...) = size(r.weight_matrix, args...)
Base.size(r::SEtoSERegridder) = (r.Nq_dst^2 * size(r.weight_matrix, 1), size(r.weight_matrix, 2))

# ──────────────────────────────────────────────────────────
# SE → FV:  dst[k] = (∑_{n∈k} W[n] * f_src[n]) / A[k]
# ──────────────────────────────────────────────────────────

function regrid!(dst_field::DenseVector, regridder::SEtoFVRegridder, src_field::DenseVector)
    LinearAlgebra.mul!(dst_field, regridder.weight_matrix, src_field)
    dst_field ./= regridder.dst_areas
    return dst_field
end

function regrid!(dst_field::AbstractVector, regridder::SEtoFVRegridder, src_field::AbstractVector)
    regridder.src_temp .= src_field
    regrid!(regridder.dst_temp, regridder, regridder.src_temp)
    dst_field .= regridder.dst_temp
    return dst_field
end

function regrid!(dst_field::DenseVector, regridder::SEtoFVRegridder, src_field::AbstractVector)
    regridder.src_temp .= src_field
    regrid!(dst_field, regridder, regridder.src_temp)
    return dst_field
end

function regrid!(dst_field::AbstractVector, regridder::SEtoFVRegridder, src_field::DenseVector)
    regrid!(regridder.dst_temp, regridder, src_field)
    dst_field .= regridder.dst_temp
    return dst_field
end

# ──────────────────────────────────────────────────────────
# FV → SE:
#   simplified: dst[n] = f_src[k]  where node n is inside cell k
#   principled: dst[n] = (Σ_k B(k,n) f_src[k]) / Wᵉᵢⱼ  (PDF Eq. 30)
# ──────────────────────────────────────────────────────────

function regrid!(dst_field::DenseVector, regridder::FVtoSERegridder, src_field::DenseVector)
    LinearAlgebra.mul!(dst_field, regridder.weight_matrix, src_field)
    if regridder.inv_node_weights !== nothing
        dst_field .*= regridder.inv_node_weights
    end
    return dst_field
end

function regrid!(dst_field::AbstractVector, regridder::FVtoSERegridder, src_field::AbstractVector)
    regridder.src_temp .= src_field
    regrid!(regridder.dst_temp, regridder, regridder.src_temp)
    dst_field .= regridder.dst_temp
    return dst_field
end

function regrid!(dst_field::DenseVector, regridder::FVtoSERegridder, src_field::AbstractVector)
    regridder.src_temp .= src_field
    regrid!(dst_field, regridder, regridder.src_temp)
    return dst_field
end

function regrid!(dst_field::AbstractVector, regridder::FVtoSERegridder, src_field::DenseVector)
    regrid!(regridder.dst_temp, regridder, src_field)
    dst_field .= regridder.dst_temp
    return dst_field
end

# ──────────────────────────────────────────────────────────
# SE → SE:  elem_vals = W * src / A_elem, then broadcast to nodes
# ──────────────────────────────────────────────────────────

function _se_to_se_regrid!(dst_field, regridder::SEtoSERegridder, src_field::DenseVector)
    N_dst_elem = length(regridder.dst_element_areas)
    Nq2 = regridder.Nq_dst^2
    elem_vals = Vector{eltype(src_field)}(undef, N_dst_elem)
    LinearAlgebra.mul!(elem_vals, regridder.weight_matrix, src_field)
    elem_vals ./= regridder.dst_element_areas
    for e in 1:N_dst_elem
        offset = (e - 1) * Nq2
        val = elem_vals[e]
        for n in 1:Nq2
            dst_field[offset + n] = val
        end
    end
    return dst_field
end

function regrid!(dst_field::DenseVector, regridder::SEtoSERegridder, src_field::DenseVector)
    _se_to_se_regrid!(dst_field, regridder, src_field)
end

function regrid!(dst_field::AbstractVector, regridder::SEtoSERegridder, src_field::AbstractVector)
    regridder.src_temp .= src_field
    _se_to_se_regrid!(dst_field, regridder, regridder.src_temp)
    return dst_field
end

function regrid!(dst_field::DenseVector, regridder::SEtoSERegridder, src_field::AbstractVector)
    regridder.src_temp .= src_field
    _se_to_se_regrid!(dst_field, regridder, regridder.src_temp)
    return dst_field
end

function regrid!(dst_field::AbstractVector, regridder::SEtoSERegridder, src_field::DenseVector)
    _se_to_se_regrid!(dst_field, regridder, src_field)
    return dst_field
end
