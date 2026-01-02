using Oceananigans
using Oceananigans.Grids: ξnode, ηnode
using Oceananigans.Fields: AbstractField
using KernelAbstractions: @index, @kernel
using ConservativeRegridding
using Statistics

instantiate(L) = L()

function compute_cell_matrix(field::AbstractField)
    Fx, Fy, _ = size(field)
    LX, LY, _ = Oceananigans.Fields.location(field)
    ℓx, ℓy = LX(), LY()

    if isnothing(ℓx) || isnothing(ℓy)
        error("cell_matrix can only be computed for fields with non-nothing horizontal location.")
    end

    grid = field.grid
    arch = grid.architecture
    FT = eltype(grid)

    vertices_per_cell = 5 # convention: [sw, nw, ne, se, sw]
    ArrayType = Oceananigans.Architectures.array_type(arch)
    cell_matrix = ArrayType{Tuple{FT, FT}}(undef, vertices_per_cell, Fx*Fy)

    arch = grid.architecture
    Oceananigans.Utils.launch!(arch, grid, (Fx, Fy), _compute_cell_matrix!, cell_matrix, Fx, ℓx, ℓy, grid)

    return cell_matrix
end

flip(::Face) = Center()
flip(::Center) = Face()

left_index(i, ::Center) = i
left_index(i, ::Face) = i-1
right_index(i, ::Center) = i + 1
right_index(i, ::Face) = i

@kernel function _compute_cell_matrix!(cell_matrix, Fx, ℓx, ℓy, grid)
    i, j = @index(Global, NTuple)

    vx = flip(ℓx)
    vy = flip(ℓy)

    isw = left_index(i, ℓx)
    jsw = left_index(j, ℓy)

    inw = left_index(i, ℓx)
    jnw = right_index(j, ℓy)

    ine = right_index(i, ℓx)
    jne = right_index(j, ℓy)

    ise = right_index(i, ℓx)
    jse = left_index(j, ℓy)

    xsw = ξnode(isw, jsw, 1, grid, vx, vy, nothing)
    ysw = ηnode(isw, jsw, 1, grid, vx, vy, nothing)

    xnw = ξnode(inw, jnw, 1, grid, vx, vy, nothing)
    ynw = ηnode(inw, jnw, 1, grid, vx, vy, nothing)

    xne = ξnode(ine, jne, 1, grid, vx, vy, nothing)
    yne = ηnode(ine, jne, 1, grid, vx, vy, nothing)

    xse = ξnode(ise, jse, 1, grid, vx, vy, nothing)
    yse = ηnode(ise, jse, 1, grid, vx, vy, nothing)

    linear_idx = i + (j - 1) * Fx
    @inbounds begin
        cell_matrix[1, linear_idx] = (xsw, ysw)
        cell_matrix[2, linear_idx] = (xnw, ynw)
        cell_matrix[3, linear_idx] = (xne, yne)
        cell_matrix[4, linear_idx] = (xse, yse)
        cell_matrix[5, linear_idx] = (xsw, ysw)
    end
end

coarse_grid = LatitudeLongitudeGrid(size=(90, 45, 1),   longitude=(0, 360), latitude=(-90, 90), z=(0, 1))
fine_grid   = LatitudeLongitudeGrid(size=(360, 180, 1), longitude=(0, 360), latitude=(-90, 90), z=(0, 1))

dst = CenterField(coarse_grid)
src = CenterField(fine_grid)

dst_cells = compute_cell_matrix(dst)
src_cells = compute_cell_matrix(src)

set!(src, (x, y, z) -> rand())

regridder = ConservativeRegridding.Regridder(dst_cells, src_cells)

ConservativeRegridding.regrid!(vec(interior(dst)), regridder, vec(interior(src)))

@test mean(dst) ≈ mean(src)

set!(dst, (x, y, z) -> rand())

ConservativeRegridding.regrid!(vec(interior(src)), transpose(regridder), vec(interior(dst)))

@test mean(dst) ≈ mean(src)

large_domain_grid = RectilinearGrid(size=(100, 100), x=(0, 2), y=(0, 2), topology=(Periodic, Periodic, Flat))
small_domain_grid = RectilinearGrid(size=(200, 200), x=(0, 1), y=(0, 1), topology=(Periodic, Periodic, Flat))

src = CenterField(small_domain_grid)
dst = CenterField(large_domain_grid)

src_cells = compute_cell_matrix(src)
dst_cells = compute_cell_matrix(dst)

set!(src, 1)

regridder = ConservativeRegridding.Regridder(dst_cells, src_cells)

ConservativeRegridding.regrid!(vec(interior(dst)), regridder, vec(interior(src)))

# Compute the integral and make sure it is the same as the original field
dst_int = Field(Integral(dst))
src_int = Field(Integral(src))

compute!(dst_int)
compute!(src_int)

@test dst_int[1, 1, 1] ≈ src[1, 1, 1]

