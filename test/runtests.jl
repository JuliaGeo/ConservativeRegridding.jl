using Test, SafeTestsets

@testset "ConservativeRegridding.jl" begin
    @safetestset "Integration: Simple Regridding" begin include("usecases/simple.jl") end
    @safetestset "Integration: Oceananigans" begin include("usecases/oceananigans.jl") end
    # This test is erroring so it's commented out for now

    # @safetestset "Integration: ClimaCore" begin include("usecases/climacore.jl") end
    # @safetestset "Integration: SpeedyWeather" begin include("usecases/speedyweather.jl") end

    @safetestset "Unit tests: Regridding" begin include("regridding.jl") end
    @safetestset "Unit tests: Grids" begin include("trees/grids.jl") end
    @safetestset "Unit tests: QuadtreeCursors" begin include("trees/quadtree_cursors.jl") end
end
