using ConservativeRegridding
using Test

@testset "Method types" begin
    @test Conservative1stOrder() isa ConservativeRegridding.AbstractRegridMethod
    @test Conservative2ndOrder() isa ConservativeRegridding.AbstractRegridMethod
end
