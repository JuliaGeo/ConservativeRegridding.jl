using ConservativeRegridding
using Test
import GeometryOps as GO
using Statistics

@testset "2nd order accuracy improvement" begin
    # Create fine source grid (10x10 cells from 11x11 points)
    src_points = [(Float64(i)/10, Float64(j)/10) for i in 0:10, j in 0:10]

    # Create coarser destination grid (5x5 cells from 6x6 points)
    dst_points = [(Float64(i)/5, Float64(j)/5) for i in 0:5, j in 0:5]

    # Linear field: f(x,y) = 2x + 3y
    # 2nd order should be exact for linear fields
    # Arrange field in column-major order (matching grid cell layout)
    src_field = Float64[]
    for j in 1:10, i in 1:10
        # Cell center
        x = (i - 0.5) / 10
        y = (j - 0.5) / 10
        push!(src_field, 2*x + 3*y)
    end

    # Expected values at destination cell centers
    expected = Float64[]
    for j in 1:5, i in 1:5
        x = (i - 0.5) / 5
        y = (j - 0.5) / 5
        push!(expected, 2*x + 3*y)
    end

    # Regrid with 1st and 2nd order
    R1 = ConservativeRegridding.Regridder(GO.Planar(), dst_points, src_points; method=Conservative1stOrder())
    R2 = ConservativeRegridding.Regridder(GO.Planar(), dst_points, src_points; method=Conservative2ndOrder())

    dst1 = zeros(25)
    dst2 = zeros(25)

    ConservativeRegridding.regrid!(dst1, R1, src_field)
    ConservativeRegridding.regrid!(dst2, R2, src_field)

    # Both should give reasonable results
    @test all(isfinite, dst1)
    @test all(isfinite, dst2)

    # 2nd order should be at least as accurate as 1st order
    error1 = mean(abs.(dst1 .- expected))
    error2 = mean(abs.(dst2 .- expected))

    @test error2 <= error1 + 1e-10  # Allow small numerical tolerance

    # For a linear field, 2nd order should be nearly exact
    @test error2 < 0.1 * error1 || error2 < 1e-10
end

@testset "Constant field preservation" begin
    # Both 1st and 2nd order should exactly preserve constant fields
    src_points = [(Float64(i)/10, Float64(j)/10) for i in 0:10, j in 0:10]
    dst_points = [(Float64(i)/5, Float64(j)/5) for i in 0:5, j in 0:5]

    constant_value = 42.0
    src_field = fill(constant_value, 100)

    R1 = ConservativeRegridding.Regridder(GO.Planar(), dst_points, src_points; method=Conservative1stOrder())
    R2 = ConservativeRegridding.Regridder(GO.Planar(), dst_points, src_points; method=Conservative2ndOrder())

    dst1 = zeros(25)
    dst2 = zeros(25)

    ConservativeRegridding.regrid!(dst1, R1, src_field)
    ConservativeRegridding.regrid!(dst2, R2, src_field)

    # Both should preserve constant exactly (within floating point tolerance)
    @test all(x -> isapprox(x, constant_value, rtol=1e-10), dst1)
    @test all(x -> isapprox(x, constant_value, rtol=1e-10), dst2)
end

@testset "Quadratic field improvement" begin
    # Quadratic field: f(x,y) = x^2 + y^2
    # 2nd order should still be better but not exact
    src_points = [(Float64(i)/10, Float64(j)/10) for i in 0:10, j in 0:10]
    dst_points = [(Float64(i)/5, Float64(j)/5) for i in 0:5, j in 0:5]

    # Create quadratic source field at cell centers
    src_field = Float64[]
    for j in 1:10, i in 1:10
        x = (i - 0.5) / 10
        y = (j - 0.5) / 10
        push!(src_field, x^2 + y^2)
    end

    # Expected values at destination cell centers
    expected = Float64[]
    for j in 1:5, i in 1:5
        x = (i - 0.5) / 5
        y = (j - 0.5) / 5
        push!(expected, x^2 + y^2)
    end

    R1 = ConservativeRegridding.Regridder(GO.Planar(), dst_points, src_points; method=Conservative1stOrder())
    R2 = ConservativeRegridding.Regridder(GO.Planar(), dst_points, src_points; method=Conservative2ndOrder())

    dst1 = zeros(25)
    dst2 = zeros(25)

    ConservativeRegridding.regrid!(dst1, R1, src_field)
    ConservativeRegridding.regrid!(dst2, R2, src_field)

    # Both should give reasonable results
    @test all(isfinite, dst1)
    @test all(isfinite, dst2)

    # Compute errors
    error1 = mean(abs.(dst1 .- expected))
    error2 = mean(abs.(dst2 .- expected))

    # 2nd order should be at least as good as 1st order
    @test error2 <= error1 + 1e-10
end

@testset "Non-nested grid improvement" begin
    # Use non-nested grids (8x8 -> 5x5) where 2nd order shows actual improvement
    # With nested grids (10x10 -> 5x5), both methods are exact for linear fields
    src_points = [(Float64(i)/8, Float64(j)/8) for i in 0:8, j in 0:8]  # 8x8 cells
    dst_points = [(Float64(i)/5, Float64(j)/5) for i in 0:5, j in 0:5]  # 5x5 cells

    # Linear field: f(x,y) = 2x + 3y
    src_field = Float64[]
    for j in 1:8, i in 1:8
        x = (i - 0.5) / 8
        y = (j - 0.5) / 8
        push!(src_field, 2*x + 3*y)
    end

    expected = Float64[]
    for j in 1:5, i in 1:5
        x = (i - 0.5) / 5
        y = (j - 0.5) / 5
        push!(expected, 2*x + 3*y)
    end

    R1 = ConservativeRegridding.Regridder(GO.Planar(), dst_points, src_points; method=Conservative1stOrder())
    R2 = ConservativeRegridding.Regridder(GO.Planar(), dst_points, src_points; method=Conservative2ndOrder())

    dst1 = zeros(25)
    dst2 = zeros(25)

    ConservativeRegridding.regrid!(dst1, R1, src_field)
    ConservativeRegridding.regrid!(dst2, R2, src_field)

    @test all(isfinite, dst1)
    @test all(isfinite, dst2)

    error1 = mean(abs.(dst1 .- expected))
    error2 = mean(abs.(dst2 .- expected))

    # 2nd order should be at least as accurate as 1st order
    @test error2 <= error1 + 1e-10

    # With non-nested grids, 2nd order should show actual improvement
    # Expect ~10-15% improvement based on testing
    @test error2 < error1
end
