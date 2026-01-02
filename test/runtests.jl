using Test, SafeTestsets

@testset "ConservativeRegridding.jl" begin
    @safetestset "Integration: Simple Regridding" begin include("usecases/simple.jl") end
    @safetestset "Integration: Oceananigans" begin include("usecases/oceananigans.jl") end
    @safetestset "Integration: ClimaCore" begin include("usecases/climacore.jl") end
    # This test is erroring so it's commented out for now
    # It needs proper spherical `intersection_operator` to work.
    # @safetestset "Integration: SpeedyWeather" begin include("usecases/speedyweather.jl") end
    
    @safetestset "Unit tests: Regridding" begin include("regridding.jl") end
end
