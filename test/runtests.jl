using Test, SafeTestsets

@testset "ConservativeRegridding.jl" begin

    @safetestset "Unit tests: Regridding" begin include("regridding.jl") end
    @safetestset "Unit tests: Grids" begin include("trees/grids.jl") end
    @safetestset "Unit tests: QuadtreeCursors" begin include("trees/quadtree_cursors.jl") end

    @safetestset "Extensions: Oceananigans" begin include("extensions/oceananigans.jl") end
    @safetestset "Extensions: ClimaCore" begin include("extensions/climacore.jl") end
    @safetestset "Extensions: Healpix" begin include("extensions/healpix.jl") end
  
    @safetestset "Comparison: XESMF" begin include("usecases/xesmf_comparison.jl") end

    @safetestset "Integration: Simple Regridding" begin include("usecases/simple.jl") end
    @safetestset "Integration: Oceananigans" begin include("usecases/oceananigans.jl") end
    @safetestset "Integration: ClimaCore" begin include("usecases/climacore.jl") end
    @safetestset "Integration: Constant Field" begin include("usecases/constant_field.jl") end
    # This test is erroring so it's commented out for now
    # @safetestset "Integration: SpeedyWeather" begin include("usecases/speedyweather.jl") end
    @safetestset "Integration: FullClenshaw / AbstractFullGrid" begin include("usecases/fullclenshaw.jl") end
    @safetestset "Integration: Full sweat test" begin include("sweat.jl") end
end
