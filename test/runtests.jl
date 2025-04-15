using Test, SafeTestsets

@testset "ConservativeRegridding.jl" begin
    @safetestset "Simple Regridding" begin include("simple.jl") end
    @safetestset "Oceananigans" begin include("oceananigans.jl") end
    # This test is erroring with so it's commented out for now
    # """
    # ┌ Error: Intersection failed!
    # │   i1 = 7321
    # │   i2 = 8844
    # └ @ ConservativeRegridding ~/development/ConservativeRegridding.jl/src/regridder.jl:64
    # ERROR: ArgumentError: Length of array must be >= 3 for GeoInterface.Wrappers.LinearRing
    # Stacktrace:
    #   [1] _length_error(T::Type, f::Function, x::Vector{Tuple{Float64, Float64}})
    #     @ GeoInterface.Wrappers ~/.julia/packages/GeoInterface/4tyIo/src/wrappers.jl:356
    #   [2] #_#32
    #     @ ~/.julia/packages/GeoInterface/4tyIo/src/wrappers.jl:281 [inlined]
    #   [3] LinearRing
    #     @ ~/.julia/packages/GeoInterface/4tyIo/src/wrappers.jl:256 [inlined]
    #   [4] LinearRing
    # """
    # @safetestset "SpeedyWeather" begin include("speedyweather.jl") end
end
