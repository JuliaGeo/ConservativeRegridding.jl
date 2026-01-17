"""
    abstract type ExampleFieldFunction

A supertype for example field functions that are used to provide values for testing and example purposes.

Any subtype must be callable with the signature `(f::ExampleFieldFunction)(lon, lat)` and return a scalar value.

See also: [`LongitudeField`](@ref), [`SinusoidField`](@ref), [`HarmonicField`](@ref), [`GulfStreamField`](@ref), [`VortexField`](@ref).
"""
abstract type ExampleFieldFunction end

"""
    LongitudeField() <: ExampleFieldFunction
    (::LongitudeField)(lon, lat)

Return the longitude of the input.  
This is a useful test for sanity checking.

This field was implemented on the advice of Simone Silvestri.
"""
struct LongitudeField <: ExampleFieldFunction end
function (::LongitudeField)(lon, lat)
    return lon
end

"""
    SinusoidField(; dp_length = 1.2π, coef = 2, coefmult = 1) <: ExampleFieldFunction
    (f::SinusoidField)(lon, lat)

A sinusoidal field for regridding testing, adapted from [^regridding-benchmarks-paper].

[^regridding-benchmarks-paper]: Valcke S, Piacentini A, Jonville G. Benchmarking Regridding Libraries Used in Earth System Modelling. Mathematical and Computational Applications. 2022; 27(2):31. https://doi.org/10.3390/mca27020031
"""
Base.@kwdef struct SinusoidField <: ExampleFieldFunction 
    dp_length::Float64 = 1.2pi
    coef::Float64 = 2
    coefmult::Float64 = 1
end
function (f::SinusoidField)(lon, lat)
    return f.coefmult * (f.coef - cos(pi * acos(cos(deg2rad(lon)) * cos(deg2rad(lat))) / f.dp_length))
end

"""
    HarmonicField() <: ExampleFieldFunction
    (::HarmonicField)(lon, lat)

A harmonic field for regridding testing, adapted from [^regridding-benchmarks-paper].

[^regridding-benchmarks-paper]: Valcke S, Piacentini A, Jonville G. Benchmarking Regridding Libraries Used in Earth System Modelling. Mathematical and Computational Applications. 2022; 27(2):31. https://doi.org/10.3390/mca27020031
"""
struct HarmonicField <: ExampleFieldFunction end
function (::HarmonicField)(lon, lat)
    return 2 + (sin(2 * deg2rad(lat))^16) * cos(16 * deg2rad(lon))
end

"""
    GulfStreamField(; dp_length = 1.2π, coef = 2.0, gf_coef = 1.0, gf_ori_lon = -80.0, gf_ori_lat = 25.0, gf_end_lon = -1.8, gf_end_lat = 50.0, gf_dmp_lon = -25.5, gf_dmp_lat = 55.5) <: ExampleFieldFunction
    (f::GulfStreamField)(lon, lat)

A field that has a large, intense "current" type structure where the Gulf Stream is located, adapted from [^regridding-benchmarks-paper].

[^regridding-benchmarks-paper]: Valcke S, Piacentini A, Jonville G. Benchmarking Regridding Libraries Used in Earth System Modelling. Mathematical and Computational Applications. 2022; 27(2):31. https://doi.org/10.3390/mca27020031
"""
Base.@kwdef struct GulfStreamField <: ExampleFieldFunction
    dp_length::Float64 = 1.2π
    coef::Float64 = 2.0
    gf_coef::Float64 = 1.0      # Gulf Stream coefficient (0.0 = no Gulf Stream)
    gf_ori_lon::Float64 = -80.0 # Origin longitude (deg)
    gf_ori_lat::Float64 = 25.0  # Origin latitude (deg)
    gf_end_lon::Float64 = -1.8  # End point longitude (deg)
    gf_end_lat::Float64 = 50.0  # End point latitude (deg)
    gf_dmp_lon::Float64 = -25.5 # Damping point longitude (deg)
    gf_dmp_lat::Float64 = 55.5  # Damping point latitude (deg)
end
function (f::GulfStreamField)(lon, lat)
    dp_conv = π / 180.0

    # Distance from origin to end and damping points
    dr0 = sqrt(((f.gf_end_lon - f.gf_ori_lon) * dp_conv)^2 +
               ((f.gf_end_lat - f.gf_ori_lat) * dp_conv)^2)
    dr1 = sqrt(((f.gf_dmp_lon - f.gf_ori_lon) * dp_conv)^2 +
               ((f.gf_dmp_lat - f.gf_ori_lat) * dp_conv)^2)

    # Base OASIS fcos analytical function
    result = f.coef - cos(π * acos(cos(lat * dp_conv) * cos(lon * dp_conv)) / f.dp_length)

    # Normalize longitude to [-180, 180]
    gf_per_lon = lon
    if gf_per_lon > 180.0
        gf_per_lon -= 360.0
    elseif gf_per_lon < -180.0
        gf_per_lon += 360.0
    end

    # Distance and angle from Gulf Stream origin
    dx = (gf_per_lon - f.gf_ori_lon) * dp_conv
    dy = (lat - f.gf_ori_lat) * dp_conv
    dr = sqrt(dx^2 + dy^2)
    dth = atan(dy, dx)

    # Gulf Stream coefficient with damping
    dc = 1.3 * f.gf_coef
    if dr > dr0
        dc = 0.0
    elseif dr > dr1
        dc *= cos(π * 0.5 * (dr - dr1) / (dr0 - dr1))
    end

    # Add Gulf Stream term
    result += (max(1000.0 * sin(0.4 * (0.5 * dr + dth) +
               0.007 * cos(50.0 * dth) + 0.37 * π), 999.0) - 999.0) * dc

    return result
end

"""
    VortexField(; lon0_rad = 5.5, lat0_rad = 0.2, r0 = 3.0, d = 5.0, t = 6.0) <: ExampleFieldFunction
    (f::VortexField)(lon, lat)

A field that has a spiral vortex on both poles of a rotated Earth designated by the parameters `lon0_rad` and `lat0_rad`.
Adapted from [^regridding-benchmarks-paper].

[^regridding-benchmarks-paper]: Valcke S, Piacentini A, Jonville G. Benchmarking Regridding Libraries Used in Earth System Modelling. Mathematical and Computational Applications. 2022; 27(2):31. https://doi.org/10.3390/mca27020031
"""
Base.@kwdef struct VortexField <: ExampleFieldFunction
    lon0_rad::Float64 = 5.5
    lat0_rad::Float64 = 0.2
    r0::Float64 = 3.0
    d::Float64 = 5.0
    t::Float64 = 6.0
end
function (f::VortexField)(lon, lat)
    # Find the rotated long and lat of the point on (long, lat) on the sphere
    # with the pole at (f.lon0_rad, f.lat0_rad).
    sin_c, cos_c = sincos(f.lat0_rad)
    sin_lat, cos_lat = sincosd(lat)
    Trm = cos_lat * cos(deg2rad(lon) - f.lon0_rad)
    X = sin_c * Trm - cos_c * sin_lat
    Y = cos_lat * sin(deg2rad(lon) - f.lon0_rad)
    Z = sin_c * sin_lat + cos_c * Trm

    # Recover lat/long
    dlon = atan(Y, X)
    if dlon < 0
        dlon = dlon + 2π
    end
    dlat = asin(Z)

    Rho = f.r0 * cos(dlat)
    Vt = 3 * sqrt(3) / 2 / cosh(Rho)^2 * tanh(Rho)
    Omega = Rho == 0 ? 0.0 : Vt / Rho

    return 2 * (1 + tanh(Rho / f.d * sin(dlon - Omega * f.t)))
end