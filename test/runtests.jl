using Test, SafeTestsets

@testset "ConservativeRegridding.jl" begin
    @safetestset "Simple Regridding" begin include("simple.jl") end
    @safetestset "SpeedyWeather" begin include("speedyweather.jl") end
    @safetestset "Oceananigans" begin include("oceananigans.jl") end
end
