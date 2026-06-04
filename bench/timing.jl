# Shared timing helper for the benchmark drivers.
#
# `timed_summary(build)` runs the zero-arg `build` closure over `reps` independent
# repetitions; each repetition collects Chairmarks samples within a `seconds_each` budget and
# every per-construction sample time is pooled. Because a repetition always yields at least one
# sample, slow constructions still get >= `reps` independent measurements while fast ones get
# many — enough, either way, to put error bars (the inter-quartile range) on each point.
#
# Returns primitive summary stats only (median + p25/p75 + sample count + min allocs/bytes), so
# results stay trivially serializable across processes and package versions.
#
# (`using` is kept at top level, not inside a guard block: `@be` is a macro and must be in scope
# when this file is lowered. Re-`include`ing is harmless — the `using` is idempotent and the
# function is simply redefined.)

using Chairmarks
using Statistics

function timed_summary(build; reps::Int = 5, seconds_each::Float64 = 0.5)
    times = Float64[]
    best = nothing                                   # min sample, for (deterministic) allocs/bytes
    for _ in 1:reps
        b = @be build() evals = 1 seconds = seconds_each
        m = minimum(b)
        best = (isnothing(best) || m.time < best.time) ? m : best
        for smp in b.samples
            push!(times, smp.time)
        end
    end
    return (;
        time_s   = median(times),
        time_lo  = quantile(times, 0.25),
        time_hi  = quantile(times, 0.75),
        nsamples = length(times),
        allocs   = Int(best.allocs),
        bytes    = Int(best.bytes),
    )
end
